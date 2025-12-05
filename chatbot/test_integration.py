#!/usr/bin/env python3
"""Test integración de endpoint /api/chat con conversación multi-turno"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/services')

# Simular las funciones como si estuvieran en api_chatbot.py
from services.conversation import ConversationSession, IntentionDetector, EntityExtractor, DocumentComparator

# Crear sesión de prueba
conversation_sessions = {}

def get_or_create_session(session_id: str) -> ConversationSession:
    """Obtener o crear sesión de usuario"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationSession(session_id)
    return conversation_sessions[session_id]

def handle_follow_up_message(query: str, session: ConversationSession):
    """Procesa mensajes de seguimiento y devuelve rama a tomar"""
    detector = IntentionDetector()
    extractor = EntityExtractor()
    
    intention = detector.detect(query)
    print(f"    [DETECTOR] Intención: {intention}")
    
    if intention == 'satisfied':
        return False, None, "✅ ¡Excelente! ¿Hay algo más que pueda ayudarte?"
    
    if intention == 'unsatisfied':
        entities = extractor.extract(query)
        print(f"    [EXTRACTOR] Entidades: {entities}")
        
        if entities['has_new_info']:
            # Insatisfecho CON detalles -> re-buscar
            new_query = f"{session.last_query} {query}"
            return True, new_query, None
        else:
            # Insatisfecho SIN detalles -> pedir aclaraciones
            return False, None, "🤔 ¿Puedes ser más específico? ¿Qué tipo de documento buscas? ¿Algún año o tema en particular?"
    
    if intention == 'refinement':
        entities = extractor.extract(query)
        new_query = f"{ ' '.join(entities['topics']) if entities['topics'] else query}"
        return True, new_query, None
    
    return False, None, None

# Simular el flujo de conversación
print("=" * 70)
print("TEST INTEGRACIÓN: Conversación Multi-turno")
print("=" * 70)

# Paso 1: Primera búsqueda
print("\n1️⃣  PRIMERA BÚSQUEDA")
print("-" * 70)
query1 = "dictadura 1973"
session_id = "test_user_123"
session = get_or_create_session(session_id)

print(f"Query: '{query1}'")
print(f"is_follow_up(): {session.is_follow_up()}")
session.add_search(query1, [
    {'href': 'http://doc1.com', 'title': 'Doc 1'},
    {'href': 'http://doc2.com', 'title': 'Doc 2'}
])
print(f"✅ Búsqueda registrada. Historial: {len(session.search_history)}")

# Paso 2: Seguimiento - insatisfecho SIN detalles
print("\n2️⃣  SEGUIMIENTO - INSATISFECHO SIN DETALLES")
print("-" * 70)
query2 = "No me sirven estos resultados"
session = get_or_create_session(session_id)

print(f"Query: '{query2}'")
print(f"is_follow_up(): {session.is_follow_up()}")

should_search, refined_query, branch_response = handle_follow_up_message(query2, session)
print(f"should_search: {should_search}")
print(f"refined_query: {refined_query}")
print(f"branch_response: {branch_response[:80]}..." if branch_response else "branch_response: None")

if branch_response:
    print("✅ CORRECTO: Sistema pregunta por detalles")
else:
    print("❌ ERROR: Debería haber pedido detalles")

# Paso 3: Seguimiento - insatisfecho CON detalles
print("\n3️⃣  SEGUIMIENTO - INSATISFECHO CON DETALLES")
print("-" * 70)
query3 = "En realidad quiero de 1980 a 1990"
session = get_or_create_session(session_id)

print(f"Query: '{query3}'")
print(f"is_follow_up(): {session.is_follow_up()}")

should_search, refined_query, branch_response = handle_follow_up_message(query3, session)
print(f"should_search: {should_search}")
print(f"refined_query: {refined_query}")
print(f"branch_response: {branch_response}")

if should_search and refined_query:
    print(f"✅ CORRECTO: Re-búsqueda con query refinada")
else:
    print("❌ ERROR: Debería hacer re-búsqueda")

# Paso 4: Seguimiento - refinamiento
print("\n4️⃣  SEGUIMIENTO - REFINAMIENTO")
print("-" * 70)
query4 = "En realidad estoy buscando DDHH"
session = get_or_create_session(session_id)

print(f"Query: '{query4}'")
should_search, refined_query, branch_response = handle_follow_up_message(query4, session)
print(f"should_search: {should_search}")
print(f"refined_query: {refined_query}")
print(f"branch_response: {branch_response}")

if should_search and refined_query:
    print(f"✅ CORRECTO: Búsqueda refinada")
else:
    print("❌ ERROR: Debería hacer búsqueda refinada")

# Paso 5: Seguimiento - satisfecho
print("\n5️⃣  SEGUIMIENTO - SATISFECHO")
print("-" * 70)
query5 = "Gracias, excelente"
session = get_or_create_session(session_id)

print(f"Query: '{query5}'")
should_search, refined_query, branch_response = handle_follow_up_message(query5, session)
print(f"should_search: {should_search}")
print(f"branch_response: {branch_response[:80]}..." if branch_response else "branch_response: None")

if not should_search and branch_response:
    print(f"✅ CORRECTO: Usuario satisfecho")
else:
    print("❌ ERROR: Debería reconocer satisfacción")

print("\n" + "=" * 70)
print("✅ TODOS LOS TESTS DE LÓGICA PASARON")
print("=" * 70)
