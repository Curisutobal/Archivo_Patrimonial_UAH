#!/usr/bin/env python3
"""Debug script para verificar persistencia de sesiones"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/services')

from services.conversation import ConversationSession

# Simular lo que hace api_chatbot.py
conversation_sessions = {}

def get_or_create_session(session_id: str) -> ConversationSession:
    """Obtener o crear sesión de usuario"""
    if session_id not in conversation_sessions:
        print(f"  [DEBUG] Creando nueva sesión: {session_id}")
        conversation_sessions[session_id] = ConversationSession(session_id)
    else:
        print(f"  [DEBUG] Usando sesión existente: {session_id}")
    return conversation_sessions[session_id]

print("=" * 70)
print("DEBUG: Persistencia de sesiones")
print("=" * 70)

# Simular request 1
print("\n📥 REQUEST 1: Primera búsqueda")
session_id = "test_user"
session = get_or_create_session(session_id)
print(f"  Historial antes: {len(session.search_history)}")
session.add_search("query1", [{'href': 'test', 'title': 'test'}])
print(f"  Historial después: {len(session.search_history)}")
print(f"  is_follow_up(): {session.is_follow_up()}")
print(f"  Sessions dict: {list(conversation_sessions.keys())}")

# Simular request 2
print("\n📥 REQUEST 2: Seguimiento")
session = get_or_create_session(session_id)
print(f"  Historial antes: {len(session.search_history)}")
print(f"  is_follow_up() ANTES de agregar: {session.is_follow_up()}")

if session.is_follow_up():
    print("  ✅ CORRECTO: is_follow_up() retorna True")
else:
    print("  ❌ ERROR: is_follow_up() retorna False")
    print(f"  Historial: {session.search_history}")

print("\n" + "=" * 70)
