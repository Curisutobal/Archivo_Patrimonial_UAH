"""
Servicio de conversación multi-turno para el chatbot
Mantiene historial de búsquedas, detecta intención del usuario,
extrae entidades y ramifica la lógica según el contexto
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ConversationSession:
    """Gestiona el historial y contexto de una conversación con un usuario"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_query = None
        self.last_results = []
        self.search_history = []  # Lista de {'query', 'results', 'timestamp'}
        self.user_satisfaction = None  # 'satisfied', 'unsatisfied', None
        
    def add_search(self, query: str, results: List[Dict]):
        """Registra una búsqueda y sus resultados"""
        self.last_query = query
        self.last_results = results
        self.search_history.append({
            'query': query,
            'results': [{'href': r.get('href'), 'title': r.get('title')} for r in results],
            'timestamp': datetime.now().isoformat()
        })
        
    def get_previous_hrefs(self) -> set:
        """Retorna los URLs de documentos de búsquedas anteriores"""
        hrefs = set()
        for search in self.search_history[:-1]:  # Excluir la última búsqueda
            for result in search['results']:
                hrefs.add(result['href'])
        return hrefs
    
    def is_follow_up(self) -> bool:
        """Determina si es un mensaje de seguimiento (no la primera búsqueda)
        
        Retorna True si ya hay al menos una búsqueda anterior,
        indicando que este es un mensaje de seguimiento/ramificación
        """
        return len(self.search_history) >= 1


class IntentionDetector:
    """Detecta la intención del usuario en mensajes de seguimiento"""
    
    # Patrones de satisfacción (en orden de precedencia)
    SATISFACTION_PATTERNS = {
        'unsatisfied': [
            r'\b(no encuentro|no está|falta|no sirve|no es|otro|diferente|otra cosa)\b',
            r'\b(no\s+son|no\s+me|estos\s+no|eso\s+no)\b',
            r'\b(en realidad|más bien|en lugar de|debería|preferir)\b',
            r'\bno\s+(es|está|me|sirve)',
            r'\b(hm|meh|nah|nope)\b',
            r'\b(espera|wait|no|eso no)\b'
        ],
        'satisfied': [
            r'\b(gracias|thank|perfecto|excelente|genial|justo|esto es|exacto|bien|ok)\b',
            r'\bme sirve\b',
            r'\b(listo|done|bueno)\b'
        ]
    }
    
    @staticmethod
    def detect(message: str) -> str:
        """
        Detecta la intención del usuario.
        Retorna: 'satisfied', 'unsatisfied', 'refinement', 'new_search'
        
        NOTA: Chequea unsatisfied PRIMERO para evitar matches falsos
        """
        message_lower = message.lower().strip()
        
        # PRIMERO chequear insatisfacción (precedencia alta)
        for pattern in IntentionDetector.SATISFACTION_PATTERNS['unsatisfied']:
            if re.search(pattern, message_lower):
                return 'unsatisfied'
        
        # LUEGO chequear satisfacción
        for pattern in IntentionDetector.SATISFACTION_PATTERNS['satisfied']:
            if re.search(pattern, message_lower):
                return 'satisfied'
        
        # Si tiene nueva información pero no dice "no encuentro"
        if EntityExtractor.extract(message)['has_new_info']:
            return 'refinement'
        
        # Por defecto, asumir que es nueva búsqueda o sigue insatisfecho
        return 'unsatisfied'


class EntityExtractor:
    """Extrae información útil del mensaje del usuario"""
    
    @staticmethod
    def extract(message: str) -> Dict:
        """
        Extrae entidades del mensaje.
        Retorna: {
            'years': [1973, 1974, ...],
            'doc_types': ['testimonios', 'fotografías', ...],
            'topics': ['derechos humanos', 'dictadura', ...],
            'has_new_info': bool
        }
        """
        message_lower = message.lower()
        result = {
            'years': [],
            'doc_types': [],
            'topics': [],
            'has_new_info': False
        }
        
        # Extraer años (1900-2099)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', message)
        result['years'] = [int(y) for y in years]
        
        # Extraer tipos de documentos
        doc_type_keywords = {
            'testimonios': ['testimonio', 'testigo', 'declaración', 'relato'],
            'fotografías': ['foto', 'fotografía', 'imagen', 'pic', 'visual'],
            'reportes': ['reporte', 'informe', 'report', 'documento'],
            'cartas': ['carta', 'carta abierta', 'missiva'],
            'actas': ['acta', 'registro', 'protocolo'],
            'comunicados': ['comunicado', 'boletín', 'aviso']
        }
        for doc_type, keywords in doc_type_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    result['doc_types'].append(doc_type)
                    break
        
        # Extraer tópicos nuevos (palabras después de "sobre", "de", "acerca")
        topics_patterns = [
            r'(?:sobre|acerca de|de)\s+([a-záéíóú\s]+?)(?:\.|,|$)',
            r'(?:principalmente|especialmente)\s+([a-záéíóú\s]+?)(?:\.|,|$)',
        ]
        for pattern in topics_patterns:
            matches = re.findall(pattern, message_lower)
            result['topics'].extend([m.strip() for m in matches if m.strip()])
        
        # Determinar si hay información nueva
        result['has_new_info'] = bool(result['years'] or result['doc_types'] or result['topics'])
        
        return result


class DocumentComparator:
    """Compara documentos entre búsquedas para encontrar similitudes"""
    
    @staticmethod
    def find_similar(new_docs: List[Dict], previous_hrefs: set) -> Tuple[List[Dict], List[Dict]]:
        """
        Compara documentos nuevos con anteriores.
        Retorna: (documentos_nuevos, documentos_similares_encontrados_antes)
        """
        new_hrefs = {doc['href'] for doc in new_docs}
        similar_indices = [i for i, doc in enumerate(new_docs) if doc['href'] in previous_hrefs]
        
        truly_new = [doc for i, doc in enumerate(new_docs) if i not in similar_indices]
        similar = [new_docs[i] for i in similar_indices]
        
        return truly_new, similar
    
    @staticmethod
    def by_topic_similarity(docs1: List[Dict], docs2: List[Dict], threshold: float = 0.5) -> float:
        """
        Calcula similitud temática entre dos conjuntos de documentos.
        Usa palabras en los títulos como indicador.
        """
        if not docs1 or not docs2:
            return 0.0
        
        def get_words(doc_list):
            words = set()
            for doc in doc_list:
                title_words = re.findall(r'\b[a-záéíóú]{3,}\b', doc.get('title', '').lower())
                words.update(title_words)
            return words
        
        words1 = get_words(docs1)
        words2 = get_words(docs2)
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
