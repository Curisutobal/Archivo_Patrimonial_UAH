"""
API Chatbot del Archivo Patrimonial UAH - Versión Completa Mejorada
=====================================================================
Funcionalidades:
- Detección de conversaciones casuales vs búsquedas
- Respuestas sin búsqueda para saludos/despedidas
- Búsqueda semántica con embeddings
- Enlaces a documentos específicos
- Manejo de errores robusto
- Puerto 5000 (Flask) + Frontend en 8080 (Nginx)
"""

import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
import pickle
import json
import os
import traceback
import pytz

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import markdown

# Google Generative AI
import google.generativeai as genai
from dotenv import load_dotenv

# Services (patterns: Abstract Factory, Proxy, Observer, Strategy, DIP)
from services.factory import ServiceFactory
from services.events import EventBus, LoggingObserver
from services.conversation import (
    ConversationSession, 
    IntentionDetector, 
    EntityExtractorImpl,
    EntityExtractor,  # Alias para backward compatibility
    DocumentComparator,
    FuzzyEntityExtractor,  # Fuzzy matching para typos
    FuzzyDocumentComparator  # Fuzzy similarity
)

# Machine Learning
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configuración de Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENAI_AVAILABLE = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GENAI_AVAILABLE = True
    except Exception as _e:
        GENAI_AVAILABLE = False
        print("⚠️ No se pudo inicializar Gemini. Continuando sin GENAI.")

embeddings_ready = False

# Initialize services and event bus (DIP + Observer)
factory = ServiceFactory(genai, GENAI_AVAILABLE)
event_bus = EventBus()
event_bus.subscribe('chat.received', LoggingObserver('[chat.received] '))
event_bus.subscribe('chat.type_detected', LoggingObserver('[chat.type_detected] '))
event_bus.subscribe('search.done', LoggingObserver('[search.done] '))
event_bus.subscribe('response.generated', LoggingObserver('[response.generated] '))

# Almacenamiento de sesiones conversacionales (en memoria para MVP)
# En producción, usar Redis o base de datos
conversation_sessions = {}

# ============================================================================
# INYECCIÓN DE DEPENDENCIAS (DIP): Estrategias para conversación
# ============================================================================
# Instancias concretas de estrategias (intercambiables con nuevas implementaciones)
intention_detector = IntentionDetector()      # Strategy: Detección de intención
entity_extractor = FuzzyEntityExtractor()     # Strategy: Extracción con fuzzy matching (tolera typos)
document_comparator = FuzzyDocumentComparator()  # Strategy: Comparación con fuzzy matching

# Alternativas sin fuzzy (si es necesario usar):
# entity_extractor = EntityExtractorImpl()     # Extracción estricta
# document_comparator = DocumentComparator()   # Comparación estricta
document_comparator = DocumentComparator()    # Strategy: Comparación de documentos

# ============================================================================
# ============================================================================

def load_documents():
    """Carga los documentos desde clean.json"""
    try:
        with open('clean.json', 'r', encoding='utf-8', errors="ignore") as f:
            docs = json.load(f)
        print(f"✅ Documentos cargados: {len(docs)}")
        return docs
    except FileNotFoundError:
        print("❌ Archivo clean.json no encontrado")
        return []
    except Exception as e:
        print(f"❌ Error cargando documentos: {e}")
        return []

documents = load_documents()

def load_embeddings():
    """Carga embeddings desde pickle o los crea si no existen"""
    try:
        with open('embeddings_cache.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✅ Embeddings cargados: {len(embeddings)} documentos")
        return embeddings
    except FileNotFoundError:
        print("⚠️ embeddings_cache.pkl no encontrado. Creando embeddings...")
        return create_embeddings_fallback()
    except Exception as e:
        print(f"⚠️ Error cargando embeddings: {e}. Recreando...")
        return create_embeddings_fallback()

def create_embeddings_fallback():
    """Crea embeddings nuevos si no existe el cache"""
    embeddings = {}
    try:
        if not GENAI_AVAILABLE:
            print("❌ GENAI no disponible; no se pueden crear embeddings nuevos.")
            return {}
        print("🔄 Generando embeddings nuevos...")
        for idx, doc in enumerate(documents):
            text = f"{doc['title']} {doc['href']}"
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings[idx] = result['embedding']
                
                if (idx + 1) % 10 == 0:
                    print(f"   Procesados {idx + 1}/{len(documents)} documentos")
                    
            except Exception as e:
                print(f"❌ Error embedding documento {idx}: {e}")
                continue
        
        # Guardar cache
        try:
            with open('embeddings_cache.pkl', 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"💾 Cache guardado: embeddings_cache.pkl")
        except Exception as e:
            print(f"⚠️ No se pudo guardar cache: {e}")
            
        print(f"✅ Embeddings creados: {len(embeddings)} documentos")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error crítico creando embeddings: {e}")
        return {}

document_embeddings = load_embeddings()
embeddings_ready = bool(document_embeddings)

# ============================================================================
# BÚSQUEDA SEMÁNTICA
# ============================================================================

def search_documents(query, top_k=6, include_suggestions=True):
    """
    Busca documentos usando similitud semántica
    Retorna hasta top_k documentos más relevantes y sugerencias de refinamiento
    """
    try:
        # Normalizar query
        normalized_query = normalize_query(query)
        print(f"🔍 Query normalizada: '{normalized_query}'")
        
        if not GENAI_AVAILABLE or not document_embeddings:
            print("⚠️ GENAI no disponible; usando búsqueda por keywords...")
            # Fallback: búsqueda por keywords en títulos
            results = search_by_keywords(normalized_query, top_k)
            suggestions = generate_search_suggestions(query, results) if include_suggestions else []
            return results, suggestions
        
        # Generar embedding de la consulta (via factory/proxy)
        query_embedder = factory.make_query_embedding()
        query_embedding = None
        embed = query_embedder(normalized_query)
        if embed is None:
            # Fallback a búsqueda por keywords si no hay embedding
            print("⚠️ No se pudo generar embedding; usando búsqueda por keywords...")
            results = search_by_keywords(normalized_query, top_k)
            suggestions = generate_search_suggestions(query, results) if include_suggestions else []
            return results, suggestions
        else:
            query_embedding = embed

        # Calcular similitudes
        similarities = []
        for idx, doc_embedding in document_embeddings.items():
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities.append((idx, similarity))

        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Retornar top_k documentos
        results = []
        for idx, score in similarities[:top_k]:
            if idx < len(documents):
                doc = documents[idx].copy()
                doc['relevance_score'] = float(score)
                results.append(doc)

        print(f"📄 Encontrados {len(results)} documentos relevantes")
        
        # Generar sugerencias si se solicita
        suggestions = []
        if include_suggestions:
            suggestions = generate_search_suggestions(query, results)
            if suggestions:
                print(f"💡 Generadas {len(suggestions)} sugerencias de refinamiento")
        
        event_bus.publish('search.done', {'query': normalized_query, 'results_count': len(results), 'suggestions_count': len(suggestions)})
        return results, suggestions
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        traceback.print_exc()
        return [], []

# ============================================================================
# NORMALIZACIÓN DE QUERIES
# ============================================================================

def is_query_too_generic(query, min_word_length=15):
    """Detecta si una consulta es muy genérica y necesita refinamiento"""
    # Términos muy amplios que suelen dar muchos resultados poco específicos
    generic_terms = [
        'dictadura', 'gobierno', 'política', 'documentos', 'archivos',
        'historia', 'chile', 'información', 'datos', 'material',
        'fotografías', 'fotos', 'imágenes', 'derechos', 'humanos',
        'partido', 'movimiento', 'organización'
    ]
    
    query_lower = query.lower().strip()
    words = query_lower.split()
    
    # Si la consulta es muy corta (1-2 palabras) y coincide con términos genéricos
    if len(words) <= 2 and len(query_lower) < min_word_length:
        if any(term in query_lower for term in generic_terms):
            return True
    
    return False

def extract_categories_from_results(results):
    """Analiza títulos de resultados para extraer categorías/temas comunes"""
    if not results:
        return []
    
    import re
    from collections import Counter
    
    # Extraer palabras clave significativas de los títulos
    keywords = []
    for doc in results:
        title = doc.get('title', '')
        # Extraer años
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', title)
        keywords.extend(years)
        
        # Extraer palabras significativas (>3 caracteres, no stopwords)
        stopwords = {'de', 'la', 'el', 'los', 'las', 'del', 'para', 'por', 'con', 'en', 'a', 'y', 'o', 'un', 'una'}
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', title.lower())
        keywords.extend([w for w in words if w not in stopwords])
    
    # Contar frecuencias
    counter = Counter(keywords)
    # Retornar las 5 más comunes (excluyendo la consulta original)
    return [word for word, count in counter.most_common(8) if count > 1]

def generate_search_suggestions(query, results):
    """Genera sugerencias de refinamiento basadas en consulta y resultados"""
    suggestions = []
    categories = extract_categories_from_results(results)
    
    query_lower = query.lower()
    
    # Sugerencias basadas en categorías encontradas
    if categories:
        # Filtrar categorías que ya están en la query
        new_categories = [cat for cat in categories[:5] if cat not in query_lower]
        if new_categories:
            suggestions.append({
                'type': 'refine_by_category',
                'message': '🎯 **Refina tu búsqueda combinando con estos temas:**',
                'options': new_categories
            })
    
    # Detectar si hay años en los resultados
    import re
    years_in_results = set()
    for doc in results:
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', doc.get('title', ''))
        years_in_results.update(years)
    
    if years_in_results and not re.search(r'\b(19\d{2}|20\d{2})\b', query):
        sorted_years = sorted(years_in_results)[:5]
        suggestions.append({
            'type': 'add_year',
            'message': '📅 **Prueba especificar un año:**',
            'options': sorted_years
        })
    
    # Sugerencias de especificidad
    if is_query_too_generic(query):
        suggestions.append({
            'type': 'be_more_specific',
            'message': '💡 **Tu búsqueda es amplia. Prueba siendo más específico:**',
            'options': [
                f'"{query} años 70"',
                f'"{query} 1973"',
                f'"{query} documentos"',
                f'"{query} fotografías"'
            ]
        })
    
    return suggestions

def normalize_query(query):
    """Normaliza y expande consultas para mejor búsqueda"""
    import unicodedata
    
    normalized = query.lower().strip()
    
    # Remover acentos
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Mapeo de términos y abreviaturas comunes
    term_mapping = {
        'dicta': 'dictadura militar',
        'ddhh': 'derechos humanos',
        'dd.hh': 'derechos humanos',
        'dd hh': 'derechos humanos',
        'mir': 'movimiento izquierda revolucionaria',
        'pc': 'partido comunista',
        'ps': 'partido socialista',
        'pdc': 'partido democrata cristiano',
        'golpe': 'golpe estado 1973',
        'pinochet': 'dictadura militar pinochet',
        'allende': 'salvador allende',
        'aylwin': 'patricio aylwin',
        '73': '1973',
        '74': '1974',
        '75': '1975',
        '76': '1976',
        '80': '1980',
        '90': '1990',
        'fotos': 'fotografias',
        'imagenes': 'fotografias',
        'pics': 'fotografias',
    }
    
    # Aplicar mapeo
    for abbrev, full_term in term_mapping.items():
        if abbrev in normalized:
            normalized = normalized.replace(abbrev, full_term)
    
    return normalized

def search_by_keywords(query, top_k=6):
    """Búsqueda fallback por palabras clave cuando GENAI no está disponible"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_docs = []
    for idx, doc in enumerate(documents):
        title_lower = doc['title'].lower()
        # Calcular score basado en coincidencias de palabras
        title_words = set(title_lower.split())
        common_words = query_words.intersection(title_words)
        
        # Score: número de palabras en común + bonus por coincidencia exacta
        score = len(common_words)
        if query_lower in title_lower:
            score += 10  # Bonus por substring exacto
        
        if score > 0:
            doc_copy = doc.copy()
            doc_copy['relevance_score'] = float(score)
            scored_docs.append((score, idx, doc_copy))
    
    # Ordenar por score descendente
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Retornar top_k
    results = [doc for _, _, doc in scored_docs[:top_k]]
    print(f"📄 Encontrados {len(results)} documentos por keywords")
    return results
    
    # Remover acentos
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # Mapeo de términos y abreviaturas comunes
    term_mapping = {
        'dicta': 'dictadura militar',
        'ddhh': 'derechos humanos',
        'dd.hh': 'derechos humanos',
        'dd hh': 'derechos humanos',
        'mir': 'movimiento izquierda revolucionaria',
        'pc': 'partido comunista',
        'ps': 'partido socialista',
        'pdc': 'partido democrata cristiano',
        'golpe': 'golpe estado 1973',
        'pinochet': 'dictadura militar pinochet',
        'allende': 'salvador allende',
        'aylwin': 'patricio aylwin',
        '73': '1973',
        '74': '1974',
        '75': '1975',
        '76': '1976',
        '80': '1980',
        '90': '1990',
        'fotos': 'fotografias',
        'imagenes': 'fotografias',
        'pics': 'fotografias',
    }
    
    # Aplicar mapeo
    for abbrev, full_term in term_mapping.items():
        if abbrev in normalized:
            normalized = normalized.replace(abbrev, full_term)
    
    return normalized

# ============================================================================
# DETECCIÓN DE TIPO DE CONVERSACIÓN
# ============================================================================

def detect_conversation_type(query):
    """
    Detecta el tipo de conversación del usuario
    Retorna: 'greeting', 'farewell', 'gratitude', 'help', 'smalltalk', 'search'
    """
    query_lower = query.lower().strip()
    
    # Eliminar puntuación
    query_clean = re.sub(r'[¿?¡!.,;:]', '', query_lower)
    
    # Patrones de saludos
    greetings = [
        r'^(hola|hello|hi|hey|ola)\s*(,|buenas?|días?|dias?|noches?|tardes?)?$',  # "hola", "hola buenas", "hola días"
        r'^(buenas|buenos)\s*(noches?|días?|dias?|tardes?)?$',  # "buenas", "buenas noches"
        r'^buen(os|as)?\s+(día|dia|días|dias|tarde|tardes|noche|noches)$',
        r'^(qué|que)\s+tal$',
        r'^cómo\s+(estás|estas|está|esta)$',
        r'^saludos$',
        r'^\s*(hola\s+)?buen(os|as)?\s*$',  # "buenos" o "hola buenos"
    ]
    
    # Patrones de despedida
    farewells = [
        r'\b(adiós|adios|chao|chau|bye|hasta\s+luego|nos\s+vemos)\b',
        r'\bgracias?\s+(por\s+todo|y\s+adiós|y\s+adios)\b',
        r'^(chao|adios|adiós|bye)$',
    ]
    
    # Patrones de agradecimiento
    gratitude = [
        r'^(gracias|muchas\s+gracias|mil\s+gracias)$',
        r'^(gracias|thank\s+you)$',
        r'^(excelente|genial|perfecto|muy\s+bien)$',
        r'^\bte\s+agradezco\b$',
    ]
    
    # Patrones de ayuda
    help_patterns = [
        r'\b(ayuda|ayudar|ayúdame|help)\b',
        r'^(qué|que)\s+(puedes?|hace|ofrece|tiene)$',
        r'^cómo\s+(funciona|usar|buscar|te\s+uso)$',
        r'^(información|info|explica|cuéntame)\s+(sobre|del|de)?\s*(bot|chatbot|ti)?$',
        r'^qué\s+es\s+esto$',
        r'^para\s+qué\s+sirve',
    ]
    
    # Patrones de smalltalk
    smalltalk = [
        r'^(cómo|como)\s+(te\s+llamas?|eres|funcionas)$',
        r'^(quién|quien)\s+eres$',
        r'^(qué|que)\s+(eres|haces)$',
        r'^\bestás\s+(bien|ahí)$',
        r'^eres\s+(un\s+)?(bot|robot|ia|inteligencia)$',
    ]
    
    # Verificar cada categoría
    for pattern in greetings:
        if re.search(pattern, query_clean):
            return 'greeting'
    
    for pattern in farewells:
        if re.search(pattern, query_clean):
            return 'farewell'
    
    for pattern in gratitude:
        if re.search(pattern, query_clean):
            return 'gratitude'
    
    for pattern in help_patterns:
        if re.search(pattern, query_clean):
            return 'help'
    
    for pattern in smalltalk:
        if re.search(pattern, query_clean):
            return 'smalltalk'
    
    # Palabras casuales cortas
    words = query_clean.split()
    if len(words) <= 2 and len(query_clean) < 15:
        casual_words = ['ok', 'vale', 'ya', 'si', 'sí', 'no', 'bien', 'mal', 'bueno']
        if any(word in casual_words for word in words):
            return 'smalltalk'
    
    # Por defecto, es una búsqueda
    return 'search'

# ============================================================================
# RESPUESTAS CONVERSACIONALES
# ============================================================================

def generate_conversational_response(query, conversation_type):
    """Genera respuestas naturales según el tipo de conversación"""
    
    # Obtener hora en zona horaria de Chile (Santiago)
    chile_tz = pytz.timezone('America/Santiago')
    hour = datetime.now(chile_tz).hour
    time_greeting = "Buenos días" if 5 <= hour < 12 else "Buenas tardes" if 12 <= hour < 20 else "Buenas noches"
    
    responses = {
        'greeting': [
            f"¡{time_greeting}! 👋 Soy el asistente del Archivo Patrimonial de la Universidad Alberto Hurtado.\n\n¿En qué puedo ayudarte hoy? Puedo ayudarte a:\n\n• 📚 Buscar documentos históricos sobre Chile\n• 📸 Explorar archivos sobre dictadura y DDHH\n• 🗂️ Encontrar material sobre movimientos sociales\n• 🏛️ Descubrir fotografías del patrimonio chileno\n\n💡 **Prueba preguntar**: \"Busca documentos sobre la dictadura\" o \"Fotografías del programa Padres e Hijos\"",
            
            f"¡Hola! 😊 Bienvenido/a al Archivo Patrimonial UAH.\n\nSoy tu asistente especializado en documentos históricos. Puedo ayudarte a explorar:\n\n📚 Historia política y social de Chile\n📸 Fotografías del programa Padres e Hijos (1974-1976)\n📄 Documentos sobre dictadura y democracia\n🗂️ Material de organizaciones sociales y DDHH\n\n¿Qué tema te gustaría explorar?",
            
            f"{time_greeting}! 🌟\n\nSoy el chatbot del Archivo Patrimonial UAH. Mi especialidad es ayudarte a encontrar documentos sobre la memoria histórica de Chile.\n\n**Puedes preguntarme cosas como:**\n• \"Busca material sobre derechos humanos\"\n• \"Documentos del MIR\"\n• \"Fotografías de los años 70\"\n• \"Material sobre la transición democrática\"\n\n¿Por dónde empezamos? 📖"
        ],
        
        'farewell': [
            "¡Hasta pronto! 👋 Fue un gusto ayudarte a explorar nuestro archivo patrimonial.\n\n📚 Recuerda que siempre puedes volver si necesitas buscar más documentos históricos.\n\n¡Que tengas un excelente día! 😊",
            
            "¡Adiós! 🌟 Espero que hayas encontrado información valiosa.\n\nVuelve cuando quieras explorar más sobre la historia y memoria de Chile. ¡Hasta luego!",
            
            "¡Nos vemos! 👋\n\nGracias por usar el Archivo Patrimonial UAH. Si necesitas más documentos históricos en el futuro, aquí estaré para ayudarte.\n\n¡Cuídate! 😊"
        ],
        
        'gratitude': [
            "¡De nada! 😊 Es un placer ayudarte a explorar nuestro patrimonio histórico.\n\n¿Hay algo más que quieras buscar en el archivo?",
            
            "¡Con gusto! 🌟 Para eso estoy aquí.\n\n¿Te gustaría explorar otros documentos o temas relacionados?",
            
            "¡Me alegra haber sido útil! 📚\n\n¿Deseas buscar más información sobre algún tema en particular?"
        ],
        
        'help': [
            """¡Claro que sí! 🤝 Te explico cómo funciono:

**🔎 ¿Qué puedo hacer?**
Busco documentos históricos del Archivo Patrimonial UAH sobre:
• Dictadura militar (1973-1990)
• Movimientos sociales y DDHH
• Partidos políticos y organizaciones
• Fotografías históricas
• Transición democrática

**💡 Ejemplos de consultas:**
• "Busca documentos sobre la dictadura militar"
• "Fotografías del programa Padres e Hijos"
• "Material sobre el MIR"
• "Documentos de derechos humanos años 80"
• "Propaganda política de los 70"

**📌 Consejos:**
✅ Usa palabras clave claras
✅ Puedo entender abreviaturas (DDHH, MIR, PC, PS)
✅ Reconozco variaciones (dictadura/dicta/DICTADURA)

**❌ Lo que NO puedo hacer:**
No manejo información sobre matrículas, horarios o temas académicos actuales (para eso visita www.uahurtado.cl)

¿Sobre qué tema histórico te gustaría buscar?""",
            
            """¡Por supuesto! 📖 Aquí te explico:

**Mi función principal:**
Soy un asistente especializado en buscar documentos del Archivo Patrimonial UAH, que contiene material histórico sobre Chile desde 1973 hasta la actualidad.

**¿Qué incluye el archivo?**
📚 Documentos políticos y sociales
📸 Fotografías históricas
📄 Material sobre dictadura y DDHH
🗂️ Testimonios y memoria colectiva
🏛️ Propaganda y documentos institucionales

**¿Cómo usarme?**
Solo escribe lo que buscas, por ejemplo:
• "derechos humanos"
• "Allende"
• "fotografías 1975"
• "movimientos sociales"

¿Qué te gustaría explorar?"""
        ],
        
        'smalltalk': [
            """¡Buena pregunta! 🤖 

Soy un asistente inteligente especializado en el **Archivo Patrimonial de la Universidad Alberto Hurtado**.

**Mi propósito:**
Ayudar a las personas a descubrir y explorar documentos históricos sobre Chile, especialmente:
• Período de dictadura militar (1973-1990)
• Movimientos sociales y DDHH
• Memoria colectiva y patrimonio cultural
• Historia política reciente

**¿Qué me hace especial?**
📚 Conozco en detalle los documentos del archivo
🔍 Puedo encontrar material específico rápidamente
💡 Entiendo contexto histórico y términos relacionados
🎯 Te ayudo a explorar temas que te interesen

¿Te gustaría que busque algo sobre la historia de Chile?""",
            
            """😊 Soy el chatbot del Archivo Patrimonial UAH.

Piensa en mí como un **bibliotecario digital especializado** que conoce muy bien el archivo y puede ayudarte a encontrar exactamente lo que buscas sobre la historia de Chile.

**Datos sobre mí:**
• Acceso a miles de documentos históricos
• Especialidad en historia reciente de Chile
• Conocimiento sobre dictadura, DDHH, movimientos sociales
• Disponible 24/7 para ayudarte

Cuéntame, ¿qué tema te interesa explorar? 📚"""
        ]
    }
    
    # Seleccionar respuesta aleatoria
    if conversation_type in responses:
        return random.choice(responses[conversation_type])
    
    return None

# ============================================================================
# GENERACIÓN DE RESPUESTAS CON IA
# ============================================================================

def generate_response(query, context_docs, suggestions=None):
    """
    Genera respuesta con IA usando Gemini
    Esta función SOLO se llama para búsquedas reales (conversation_type='search')
    Ahora incluye sugerencias de refinamiento si se proporcionan
    """
    suggestions = suggestions or []
    
    # Verificar temas administrativos (fuera de alcance)
    administrative_keywords = [
        'matricula', 'matrícula', 'inscripción', 'inscripcion',
        'horario', 'horarios', 'clases', 'notas', 'calificaciones',
        'malla', 'curricular', 'admisión', 'admision', 'postular',
        'aranceles', 'becas', 'financiamiento', 'pago', 'cuota',
        'profesor', 'docente', 'contacto', 'email', 'teléfono', 'telefono',
        'carrera', 'carreras', 'programa', 'postgrado', 'magister'
    ]
    
    query_lower = query.lower()
    is_administrative = any(keyword in query_lower for keyword in administrative_keywords)
    
    if is_administrative:
        return """🎓 Esta consulta está fuera del alcance del Archivo Patrimonial.

📚 El **Archivo Patrimonial UAH** se enfoca en documentos históricos y patrimonio cultural chileno (1973-actualidad).

Para información sobre **matrículas, horarios, admisión y temas académicos**, por favor visita:

🌐 **Sitio web oficial**: [www.uahurtado.cl](https://www.uahurtado.cl)
📱 **Instagram UAH**: [@uahurtado](https://www.instagram.com/uahurtado/)
📧 **Email**: [email protected]

---

💡 **¿Sabías que...?** Nuestro archivo contiene documentos fascinantes sobre la historia de Chile. ¿Te gustaría explorar algún tema histórico?"""
    
    # Si no hay documentos relevantes
    if not context_docs:
        return """🔍 No encontré documentos específicos para tu consulta.

**Sugerencias para mejorar tu búsqueda:**

✅ **Intenta con términos más específicos:**
• En lugar de: "información" → Prueba: "dictadura militar"
• En lugar de: "fotos" → Prueba: "fotografías programa Padres e Hijos"

✅ **Verifica la ortografía** de los términos de búsqueda

✅ **Usa palabras clave** relacionadas con:
• Dictadura militar (1973-1990)
• Derechos humanos (DDHH)
• Movimientos sociales
• Partidos políticos (MIR, PC, PS)
• Fotografías históricas
• Patricio Aylwin

💡 **Ejemplos que funcionan bien:**
• "documentos sobre la dictadura"
• "fotografías de los años 70"
• "material del MIR"
• "derechos humanos años 80"

¿Te gustaría reformular tu búsqueda?"""
    
    # Si no hay GENAI disponible, respuesta básica
    if not GENAI_AVAILABLE:
        response = "📚 **He encontrado estos documentos relevantes:**\n\n"
        for i, doc in enumerate(context_docs, 1):
            response += f"{i}. **{doc['title']}**\n"
            response += f"   🔗 [Ver documento]({doc['href']})\n\n"
        return response
    
    # Construir contexto para la IA
    try:
        context = "Documentos relevantes encontrados:\n\n"
        for i, doc in enumerate(context_docs, 1):
            score = doc.get('relevance_score', 0)
            context += f"{i}. {doc['title']} (relevancia: {score:.2f})\n"
            context += f"   URL: {doc['href']}\n\n"
        
        # Construir sección de sugerencias si existen
        suggestions_text = ""
        if suggestions:
            suggestions_text = "\n\nSUGERENCIAS DE REFINAMIENTO PARA EL USUARIO:\n"
            for sug in suggestions:
                suggestions_text += f"- {sug['message']}\n"
                if sug.get('options'):
                    for opt in sug['options'][:3]:  # Limitar a 3 opciones
                        suggestions_text += f"  • {opt}\n"
        
        prompt = f"""Eres un asistente amigable del Archivo Patrimonial UAH, especializado en documentos históricos de Chile.

DOCUMENTOS ENCONTRADOS:
{context}

CONSULTA DEL USUARIO: "{query}"{suggestions_text}

INSTRUCCIONES:
- Presenta los documentos encontrados de forma clara y organizada
- Incluye enlaces markdown: [Título del documento](URL)
- Explica brevemente la relevancia de cada documento para la consulta
- Proporciona contexto histórico cuando sea pertinente
- Usa un tono profesional pero cercano y amigable
- Usa emojis ocasionales para hacer la respuesta más visual
- Si hay SUGERENCIAS DE REFINAMIENTO, incorpóralas naturalmente al final de tu respuesta, explicando que si el usuario no encontró lo que buscaba, puede refinar con esos términos
- Al final, invita al usuario a seguir explorando o hacer más preguntas

IMPORTANTE: 
- Menciona que los enlaces llevan directamente a los documentos en el archivo
- Si algún documento es especialmente relevante, destácalo
- Si la consulta es genérica y hay sugerencias, menciona amablemente que puede ser más específico

Responde de forma natural, útil y educativa:"""

        responder = factory.make_responder()
        resp_text = responder(prompt)
        if resp_text is not None:
            return resp_text

        # Fallback: si no hay GENAI disponible, respuesta básica con sugerencias
        response = "📚 **Encontré estos documentos relevantes:**\n\n"
        for i, doc in enumerate(context_docs, 1):
            response += f"{i}. **{doc['title']}**\n"
            response += f"   🔗 [Ver documento]({doc['href']})\n\n"
        
        # Añadir sugerencias si existen
        if suggestions:
            response += "\n---\n\n"
            for sug in suggestions:
                response += f"{sug['message']}\n"
                if sug.get('options'):
                    for opt in sug['options'][:4]:
                        response += f"  • {opt}\n"
                response += "\n"
        
        response += "\n💡 **Nota:** Si estos documentos no son exactamente lo que buscabas, intenta refinar tu búsqueda con términos más específicos."
        return response
        
    except Exception as e:
        print(f"❌ Error con Gemini: {e}")
        traceback.print_exc()

        # Fallback: respuesta sin IA
        response = "📚 **Encontré estos documentos relevantes:**\n\n"
        for i, doc in enumerate(context_docs, 1):
            response += f"{i}. **{doc['title']}**\n"
            response += f"   🔗 [Ver documento]({doc['href']})\n\n"
        response += "\n💡 **Nota:** Estoy experimentando limitaciones técnicas, pero aquí están los documentos que coinciden con tu búsqueda."
        return response

# ============================================================================
# LÓGICA DE CONVERSACIÓN MULTI-TURNO
# ============================================================================

def get_or_create_session(session_id: str) -> ConversationSession:
    """Obtiene o crea una sesión conversacional para un usuario"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationSession(session_id)
    return conversation_sessions[session_id]

def handle_follow_up_message(query: str, session: ConversationSession) -> tuple:
    """
    Maneja mensajes de seguimiento (no es la primera búsqueda).
    Retorna: (debe_hacer_nueva_busqueda: bool, nueva_query: str, respuesta_ramificacion: str or None)
    
    Usa estrategias inyectadas (DIP): intention_detector, entity_extractor
    """
    # PASO 1: Eliminar saludos del mensaje para detectar intención real
    cleaned_query = intention_detector.remove_greetings(query)
    print(f"🧹 Mensaje limpiado de saludos: '{query}' → '{cleaned_query}'")
    
    # PASO 2: Detectar si hay búsqueda explícita
    has_search = intention_detector.has_explicit_search(cleaned_query)
    print(f"🔍 ¿Hay búsqueda explícita? {has_search}")
    
    # PASO 3: Si NO hay búsqueda explícita, detectar intención (satisfecho, insatisfecho, etc)
    if not has_search:
        intention = intention_detector.detect(cleaned_query)
        print(f"🎯 Intención detectada: {intention}")
        
        # Caso 1: Usuario satisfecho
        if intention == 'satisfied':
            response = "¡Excelente! 😊 Me alegra haber encontrado lo que buscabas.\n\n¿Hay algo más que quieras explorar en el Archivo Patrimonial?"
            return False, None, response
        
        # Caso 2: Usuario insatisfecho SIN información adicional
        if intention == 'unsatisfied':
            response = """❓ Entiendo que no encontraste lo que buscabas. \n\n**Para poder ayudarte mejor, ¿podrías ser más específico?** 🤔\n\n💡 Por ejemplo:\n• **Período:** ¿De qué años? (1973-1990, 1980-1985, etc.)\n• **Tipo:** ¿Fotografías, testimonios, documentos, reportes?\n• **Tema:** ¿Hay un aspecto específico? (DDHH, partido político, organización)\n• **Persona:** ¿Hay alguien específico involucrado?\n\nCuéntame más y haré una búsqueda más dirigida. 📚"""
            return False, None, response
    
    # PASO 4: Si HAY búsqueda explícita, hacer nueva búsqueda
    # (ignorar intención de satisfacción/insatisfacción si hay términos de búsqueda claros)
    entities = entity_extractor.extract(cleaned_query)
    query_parts = []
    
    if entities['topics']:
        query_parts.extend(entities['topics'])
    if entities['years']:
        query_parts.append(f"{min(entities['years'])}")
    if entities['doc_types']:
        query_parts.extend(entities['doc_types'])
    
    # Si no extrajimos nada, usar el query limpiado
    new_query = ' '.join(query_parts) if query_parts else cleaned_query
    print(f"🔍 Nueva búsqueda será: '{new_query}'")
    
    return True, new_query, None

def compare_and_format_results(new_docs: List[Dict], session: ConversationSession, original_query: str) -> str:
    """
    Compara resultados nuevos con anteriores y formatea respuesta apropiada.
    """
    truly_new, similar = DocumentComparator.find_similar(new_docs, session.get_previous_hrefs())
    
    response = ""
    
    if similar:
        response += "📌 **Documentos encontrados (algunos similares a búsquedas anteriores):**\n\n"
        for i, doc in enumerate(new_docs, 1):
            is_similar = "🔄 " if doc in similar else "✨ "
            response += f"{is_similar}{i}. **{doc['title']}**\n"
            response += f"   🔗 [Ver documento]({doc['href']})\n\n"
    else:
        response += "📚 **Aquí están los documentos encontrados con tu búsqueda refinada:**\n\n"
        for i, doc in enumerate(new_docs, 1):
            response += f"{i}. **{doc['title']}**\n"
            response += f"   🔗 [Ver documento]({doc['href']})\n\n"
    
    # Calcular similitud temática
    topic_similarity = DocumentComparator.by_topic_similarity(
        session.last_results,
        new_docs
    )
    
    if truly_new:
        response += f"\n💡 Encontré **{len(truly_new)} documento(s) nuevo(s)** en esta búsqueda que no aparecían antes.\n"
    
    if topic_similarity > 0.6:
        response += f"🔗 Estos resultados tienen alta similitud temática con tu búsqueda anterior.\n"
    elif topic_similarity > 0.3:
        response += f"🔄 Estos resultados comparten algunos temas con la búsqueda anterior.\n"
    else:
        response += f"🆕 Estos resultados son bastante diferentes a los anteriores.\n"
    
    response += "\n**¿Encontraste lo que buscabas?** Si no, cuéntame más y seguiremos buscando. 😊"
    
    return response

# ============================================================================
# RUTAS DE LA API
# ============================================================================


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """
    Endpoint principal del chatbot
    Detecta tipo de conversación y responde apropiadamente
    """
    # Manejar preflight CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    query = ""
    session_id = ""
    try:
        print(f"\n{'='*60}")
        print(f"📥 Nueva solicitud recibida")
        print(f"{'='*60}")
        print(f"🔥 Método: {request.method}")
        print(f"🔥 Content-Type: {request.content_type}")

        # Obtener query de JSON o form
        data = request.get_json(silent=True)
        if data and isinstance(data, dict):
            query = data.get('query', '')
            session_id = data.get('session_id', 'default')
            print(f"✅ Query from JSON: '{query}'")
        else:
            query = request.form.get('query', '')
            session_id = request.form.get('session_id', 'default')
            print(f"✅ Query from form: '{query}'")

        query = query.strip()
        print(f"🔎 Query procesada: '{query}'")
        print(f"🆔 Session ID: {session_id}")
        event_bus.publish('chat.received', {'query': query})

        if not query:
            print("❌ Query vacía")
            return jsonify({
                'success': False,
                'error': 'No query provided', 
                'details': 'La consulta no puede estar vacía.'
            }), 400

        # Obtener o crear sesión del usuario
        session = get_or_create_session(session_id)
        print(f"📋 Session creada/recuperada. Historial: {len(session.search_history)} búsquedas")

        # ============ PASO 1: DETECTAR TIPO DE CONVERSACIÓN PRIMERO ============
        # Esto asegura que saludos, despedidas, etc. se respondan conversacionalmente
        # INCLUSO en mensajes de seguimiento (si es solo "Hola buenas", no debe hacer búsqueda)
        conversation_type = detect_conversation_type(query)
        print(f"🎯 Tipo detectado: {conversation_type}")
        event_bus.publish('chat.type_detected', {'type': conversation_type})
        
        # ============ PASO 2: SI ES CONVERSACIÓN CASUAL, RESPONDER SIN BUSCAR ============
        if conversation_type in ['greeting', 'farewell', 'gratitude', 'help', 'smalltalk']:
            print(f"💬 Respuesta conversacional (sin búsqueda)")
            response_text = generate_conversational_response(query, conversation_type)
            
            if response_text:
                response_html = markdown.markdown(response_text)
                event_bus.publish('response.generated', {'chars': len(response_text), 'docs': 0})
                
                return jsonify({
                    'success': True,
                    'response': response_html,
                    'documents': [],
                    'embeddings_ready': embeddings_ready,
                    'conversation_type': conversation_type,
                    'session_id': session_id
                })

        # Verificar si es seguimiento ANTES de agregar la búsqueda actual
        is_follow_up = session.is_follow_up()
        
        # ============ LÓGICA DE SEGUIMIENTO ============
        # Si es un mensaje de seguimiento (no la primera búsqueda)
        if is_follow_up:
            print(f"🔄 Mensaje de seguimiento detectado (búsquedas previas: {len(session.search_history)})")
            should_search, refined_query, branch_response = handle_follow_up_message(query, session)
            
            if branch_response:
                # Es una ramificación de la lógica (satisfecho, pedir detalles, etc)
                print(f"💬 Respuesta de ramificación: {len(branch_response)} caracteres")
                response_html = markdown.markdown(branch_response)
                event_bus.publish('response.generated', {'chars': len(branch_response), 'docs': 0})
                
                return jsonify({
                    'success': True,
                    'response': response_html,
                    'documents': [],
                    'embeddings_ready': embeddings_ready,
                    'conversation_type': 'follow_up_branch',
                    'session_id': session_id
                })
            
            if not should_search:
                # No hay nueva búsqueda que hacer
                return jsonify({
                    'success': True,
                    'response': markdown.markdown("✅ ¿En qué más te puedo ayudar?"),
                    'documents': [],
                    'embeddings_ready': embeddings_ready,
                    'conversation_type': 'follow_up_satisfied',
                    'session_id': session_id
                })
            
            # Hay nueva búsqueda que hacer
            query = refined_query
            print(f"🔍 Nueva búsqueda con query refinada: '{query}'")

        # ============ FLUJO NORMAL (primera búsqueda o búsqueda refinada) ============
        # PASO 3: SI ES 'search', BUSCAR DOCUMENTOS (6 documentos) Y SUGERENCIAS
        print(f"🔍 Realizando búsqueda de documentos...")
        relevant_docs, suggestions = search_documents(query, top_k=6, include_suggestions=True)
        print(f"📄 Encontrados {len(relevant_docs)} documentos")
        if suggestions:
            print(f"💡 Generadas {len(suggestions)} sugerencias")
        
        # Registrar búsqueda en la sesión
        session.add_search(query, relevant_docs)

        # PASO 4: GENERAR RESPUESTA CON IA (incluyendo sugerencias)
        print(f"🤖 Generando respuesta con IA...")
        response_text = generate_response(query, relevant_docs, suggestions)
        print(f"✅ Respuesta generada: {len(response_text)} caracteres")

        # Convertir markdown a HTML
        response_html = markdown.markdown(response_text)
        event_bus.publish('response.generated', {'chars': len(response_text), 'docs': len(relevant_docs)})

        return jsonify({
            'success': True,
            'response': response_html,
            'documents': relevant_docs,
            'embeddings_ready': embeddings_ready,
            'conversation_type': conversation_type,
            'session_id': session_id
        })

    except Exception as e:
        print(f"\n❌ ERROR EN /chat:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': False,
            'error': 'Error procesando la consulta',
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de health check"""
    return jsonify({
        'status': 'ok',
        'documents_loaded': len(documents),
        'embeddings_loaded': len(document_embeddings),
        'genai_available': GENAI_AVAILABLE
    })

@app.route('/', methods=['GET'])
def index():
    """Ruta raíz"""
    return jsonify({
        'message': 'API Chatbot Archivo Patrimonial UAH',
        'version': '2.0',
        'endpoints': {
            'chat': '/api/chat (POST)',
            'health': '/api/health (GET)'
        }
    })

# ============================================================================
# MAIN: EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 INICIANDO CHATBOT DEL ARCHIVO PATRIMONIAL UAH")
    print("="*70)
    print(f"📊 Documentos cargados: {len(documents)}")
    print(f"🧠 Embeddings disponibles: {len(document_embeddings)}")
    print(f"🤖 Gemini API: {'✅ Disponible' if GENAI_AVAILABLE else '❌ No disponible'}")
    print(f"🌐 Servidor Flask: http://localhost:5000")
    print(f"🔗 Frontend esperado: http://localhost:8080 (vía Nginx)")
    print(f"📡 Endpoint principal: POST /api/chat")
    print(f"❤️ Health check: GET /api/health")
    print("="*70)
    print("✅ Sistema listo para recibir consultas!\n")

    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )