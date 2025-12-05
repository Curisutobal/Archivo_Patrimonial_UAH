"""
Servicio de conversación multi-turno para el chatbot
Mantiene historial de búsquedas, detecta intención del usuario,
extrae entidades y ramifica la lógica según el contexto

Principios SOLID implementados:
- SRP: Cada clase tiene una responsabilidad única
- OCP: Extensible mediante estrategias base (IntentionStrategy, EntityStrategy)
- ISP: Interfaces pequeñas y específicas
- DIP: Inyección de dependencias

Patrones de Diseño:
- Strategy Pattern: Estrategias intercambiables para detección
- Decorator Pattern: Metadata envolviendo resultados de búsqueda
- Factory Pattern: ConversationManager como factory
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# ABSTRACCIONES (OCP + DIP): Permitir extensión sin modificación
# ============================================================================

class IntentionStrategy(ABC):
    """Abstracción para estrategias de detección de intención (Strategy Pattern)
    
    Principio OCP: Open/Closed - Extensible sin modificar código existente
    Principio DIP: Dependency Inversion - Dependemos de abstractos, no de concretos
    """
    
    @abstractmethod
    def detect(self, message: str) -> str:
        """
        Detecta intención del usuario.
        Retorna: 'satisfied', 'unsatisfied', 'refinement', 'new_search'
        """
        pass


class EntityStrategy(ABC):
    """Abstracción para estrategias de extracción de entidades (Strategy Pattern)
    
    Principio OCP: Open/Closed - Extensible sin modificar código existente
    Principio DIP: Dependency Inversion - Dependemos de abstractos, no de concretos
    """
    
    @abstractmethod
    def extract(self, message: str) -> Dict:
        """
        Extrae entidades del mensaje.
        Retorna: {'years': [...], 'doc_types': [...], 'topics': [...], 'has_new_info': bool}
        """
        pass


class SimilarityStrategy(ABC):
    """Abstracción para estrategias de comparación de documentos
    
    Principio OCP: Open/Closed - Extensible sin modificar código existente
    """
    
    @abstractmethod
    def find_similar(self, new_docs: List[Dict], previous_hrefs: set) -> Tuple[List[Dict], List[Dict]]:
        """Compara documentos nuevos con anteriores"""
        pass
    
    @abstractmethod
    def calculate_topic_similarity(self, docs1: List[Dict], docs2: List[Dict]) -> float:
        """Calcula similitud temática entre conjuntos de documentos"""
        pass


# ============================================================================
# IMPLEMENTACIONES CONCRETAS CON INYECCIÓN DE DEPENDENCIAS
# ============================================================================

class ConversationSession:
    """Gestiona el historial y contexto de una conversación (SRP)
    
    Principio SRP: Solo gestiona sesiones y su historial
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_query = None
        self.last_results = []
        self.search_history = []
        self.user_satisfaction = None
        
    def add_search(self, query: str, results: List[Dict]):
        """Registra una búsqueda (Decorator Pattern - envuelve con metadata)"""
        self.last_query = query
        self.last_results = results
        self.search_history.append({
            'query': query,
            'results': [{'href': r.get('href'), 'title': r.get('title')} for r in results],
            'timestamp': datetime.now().isoformat()
        })
        
    def get_previous_hrefs(self) -> set:
        """Retorna URLs de búsquedas anteriores"""
        hrefs = set()
        for search in self.search_history[:-1]:
            for result in search['results']:
                hrefs.add(result['href'])
        return hrefs
    
    def is_follow_up(self) -> bool:
        """¿Es un mensaje de seguimiento?"""
        return len(self.search_history) >= 1


class IntentionDetector(IntentionStrategy):
    """Detección de intención mediante regex (SRP + Strategy Pattern)
    
    Principio SRP: Solo detecta intención
    Implementa: IntentionStrategy (polimorfismo)
    DIP: Patrones inyectables en __init__
    """
    
    def __init__(self, patterns: Optional[Dict[str, List[str]]] = None):
        """Inyección de dependencias (DIP)
        
        Args:
            patterns: Dict con patrones regex personalizados
        """
        self.patterns = patterns or self._default_patterns()
    
    @staticmethod
    def _default_patterns() -> Dict[str, List[str]]:
        """Patrones por defecto (separados del constructor para OCP)"""
        return {
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
    
    def detect(self, message: str) -> str:
        """Implementación de IntentionStrategy (polimorfismo)"""
        message_lower = message.lower().strip()
        
        # PRIMERO chequear insatisfacción (precedencia alta)
        for pattern in self.patterns['unsatisfied']:
            if re.search(pattern, message_lower):
                return 'unsatisfied'
        
        # LUEGO chequear satisfacción
        for pattern in self.patterns['satisfied']:
            if re.search(pattern, message_lower):
                return 'satisfied'
        
        # Si tiene nueva información
        extractor = EntityExtractorImpl()
        if extractor.extract(message).get('has_new_info'):
            return 'refinement'
        
        return 'unsatisfied'


class EntityExtractorImpl(EntityStrategy):
    """Extracción de entidades mediante regex (SRP + Strategy Pattern)
    
    Principio SRP: Solo extrae entidades
    Implementa: EntityStrategy (polimorfismo)
    DIP: Tipos de documentos inyectables
    """
    
    def __init__(self, doc_types: Optional[Dict[str, List[str]]] = None):
        """Inyección de dependencias (DIP)
        
        Args:
            doc_types: Dict personalizado de tipos de documentos
        """
        self.doc_types = doc_types or self._default_doc_types()
    
    @staticmethod
    def _default_doc_types() -> Dict[str, List[str]]:
        """Tipos por defecto (separados para OCP)"""
        return {
            'testimonios': ['testimonio', 'testigo', 'declaración', 'relato'],
            'fotografías': ['foto', 'fotografía', 'imagen', 'pic', 'visual'],
            'reportes': ['reporte', 'informe', 'report', 'documento'],
            'cartas': ['carta', 'carta abierta', 'missiva'],
            'actas': ['acta', 'registro', 'protocolo'],
            'comunicados': ['comunicado', 'boletín', 'aviso']
        }
    
    def extract(self, message: str) -> Dict:
        """Implementación de EntityStrategy"""
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
        for doc_type, keywords in self.doc_types.items():
            for keyword in keywords:
                if keyword in message_lower:
                    result['doc_types'].append(doc_type)
                    break
        
        # Extraer tópicos
        topics_patterns = [
            r'(?:sobre|acerca de|de)\s+([a-záéíóú\s]+?)(?:\.|,|$)',
            r'(?:principalmente|especialmente)\s+([a-záéíóú\s]+?)(?:\.|,|$)',
        ]
        for pattern in topics_patterns:
            matches = re.findall(pattern, message_lower)
            result['topics'].extend([m.strip() for m in matches if m.strip()])
        
        result['has_new_info'] = bool(result['years'] or result['doc_types'] or result['topics'])
        
        return result


# Alias para compatibilidad backward
EntityExtractor = EntityExtractorImpl


class DocumentComparator(SimilarityStrategy):
    """Comparación de documentos (SRP + Strategy Pattern)
    
    Principio SRP: Solo compara documentos
    Implementa: SimilarityStrategy (polimorfismo)
    """
    
    def find_similar(self, new_docs: List[Dict], previous_hrefs: set) -> Tuple[List[Dict], List[Dict]]:
        """Implementación de SimilarityStrategy"""
        new_hrefs = {doc['href'] for doc in new_docs}
        similar_indices = [i for i, doc in enumerate(new_docs) if doc['href'] in previous_hrefs]
        
        truly_new = [doc for i, doc in enumerate(new_docs) if i not in similar_indices]
        similar = [new_docs[i] for i in similar_indices]
        
        return truly_new, similar
    
    def calculate_topic_similarity(self, docs1: List[Dict], docs2: List[Dict], threshold: float = 0.5) -> float:
        """Implementación de SimilarityStrategy"""
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


# Alias para compatibilidad backward
by_topic_similarity = lambda docs1, docs2, threshold=0.5: DocumentComparator().calculate_topic_similarity(docs1, docs2, threshold)
