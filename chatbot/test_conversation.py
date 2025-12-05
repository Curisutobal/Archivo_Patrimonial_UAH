#!/usr/bin/env python3
"""Test script para verificar que las clases de conversación funcionen"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/services')

from services.conversation import ConversationSession, IntentionDetector, EntityExtractor

print("=" * 60)
print("TEST 1: is_follow_up()")
print("=" * 60)

session = ConversationSession('test_session')
print(f"is_follow_up() con historial vacío: {session.is_follow_up()}")

session.add_search('test query', [{'href': 'http://test.com', 'title': 'Test'}])
print(f"is_follow_up() con 1 búsqueda: {session.is_follow_up()}")

session.add_search('test query 2', [{'href': 'http://test2.com', 'title': 'Test 2'}])
print(f"is_follow_up() con 2 búsquedas: {session.is_follow_up()}")

print("\n" + "=" * 60)
print("TEST 2: IntentionDetector")
print("=" * 60)

detector = IntentionDetector()
test_messages = [
    "No me sirven",
    "Gracias",
    "Quiero derechos humanos en 1975",
    "En realidad busco algo diferente"
]

for msg in test_messages:
    intention = detector.detect(msg)
    print(f"'{msg}' -> {intention}")

print("\n" + "=" * 60)
print("TEST 3: EntityExtractor")
print("=" * 60)

extractor = EntityExtractor()
test_messages_with_entities = [
    "Busco documentos de 1975 sobre dictadura",
    "Quiero comunicados de 1980 a 1990",
    "Manifestos de derechos humanos"
]

for msg in test_messages_with_entities:
    entities = extractor.extract(msg)
    print(f"'{msg}'")
    print(f"  -> {entities}")

print("\n✅ Todos los tests completados")
