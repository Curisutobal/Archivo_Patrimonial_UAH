# 🎯 Implementación: Conversación Multi-turno con Ramificaciones

**Fecha**: 5 de Diciembre de 2025  
**Estado**: ✅ COMPLETADO Y TESTEADO

---

## 📋 Resumen Ejecutivo

Se ha implementado un sistema de **conversación multi-turno** que mantiene contexto entre múltiples mensajes del usuario, detecta su intención (satisfecho/insatisfecho/refinamiento) y ramifica la lógica de búsqueda dinámicamente.

**Características principales**:
- ✅ Gestión de sesiones por usuario (`session_id`)
- ✅ Detección de intención del usuario (4 tipos)
- ✅ Extracción de entidades (años, tipos de doc, tópicos)
- ✅ Comparación de documentos (similares vs nuevos)
- ✅ Ramificación lógica según contexto
- ✅ Conversación sin estado desaparecida; ahora es **stateful**

---

## 🏗️ Arquitectura Implementada

### 1. **Capa de Conversación** (`chatbot/services/conversation.py`)

#### Clases:

**`ConversationSession`**
```python
- session_id: str                          # ID único por usuario
- search_history: list[dict]              # Historial de búsquedas {query, results, timestamp}
- last_query: str                         # Última query
- last_results: list[dict]               # Últimos resultados

Métodos:
- add_search(query, results)             # Registra búsqueda
- get_previous_hrefs()                    # URLs de búsquedas anteriores
- is_follow_up()                          # ¿Es seguimiento? (>= 1 búsqueda anterior)
```

**`IntentionDetector`**
```python
Detecta intención del usuario en mensajes de seguimiento:
- 'satisfied'    → Usuario contento (keywords: "gracias", "excelente", "perfecto")
- 'unsatisfied'  → Usuario insatisfecho (keywords: "no me sirven", "no encuentro")
- 'refinement'   → Cambio/refinamiento (keywords: "en realidad", "quiero cambiar")
- 'new_search'   → Nueva búsqueda (by default si no match)

Implementación: Regex + keyword patterns (extensible a Gemini)
```

**`EntityExtractor`**
```python
Extrae entidades de mensajes del usuario:
- years: list[int]              # Años (1900-2099)
- doc_types: list[str]          # Tipos: comunicados, reportes, etc
- topics: list[str]             # Tópicos: derechos humanos, dictadura, etc
- has_new_info: bool            # ¿Hay información nueva en el mensaje?

Implementación: Regex patterns + keyword matching
```

**`DocumentComparator`**
```python
Compara documentos entre búsquedas:
- find_similar(new_docs, previous_hrefs) → (truly_new, similar)
- by_topic_similarity(new_docs, prev_results) → overlap score

Usa URLs exactas y similitud de títulos (palabras en común)
```

### 2. **Lógica de Ramificación** (en `api_chatbot.py`)

#### Funciones:

**`get_or_create_session(session_id)`**
- Obtiene sesión existente del dict `conversation_sessions`
- O crea una nueva si no existe

**`handle_follow_up_message(query, session)`**
```
Retorna: (should_search: bool, refined_query: str, branch_response: str)

Lógica:
- Detecta intención → EntityExtractor extrae info
- Si satisfied: return (False, None, "¡Excelente!")
- Si unsatisfied:
  - CON detalles → return (True, refined_query, None)  # Re-buscar
  - SIN detalles → return (False, None, "¿Puedes ser más específico?")
- Si refinement → return (True, new_query, None)
```

**`compare_and_format_results(new_docs, session, original_query)`**
- Compara resultados nuevos vs anteriores
- Marca documentos repetidos con 🔄
- Marca nuevos con ✨
- Formatea respuesta con contexto de búsqueda

### 3. **Integración en Endpoint** (`/api/chat`)

```
Flujo mejorado:

1. Extraer session_id del request (body JSON)
2. get_or_create_session(session_id) → session
3. Verificar is_follow_up() ANTES de agregar búsqueda
   ├─ Si FALSE (primera búsqueda):
   │  └─ detect_conversation_type() → search/greeting/etc
   │     └─ search_documents() → results
   │        └─ session.add_search(query, results)
   │           └─ Retornar con documentos
   │
   └─ Si TRUE (seguimiento):
      └─ handle_follow_up_message(query, session)
         ├─ Si branch_response: Retornar ramificación (pedir detalles, etc)
         ├─ Si should_search=True: Hacer search_documents(refined_query)
         │  └─ session.add_search(query, results)
         │     └─ compare_and_format_results() → response
         └─ Si should_search=False: Retornar fin de conversación
```

---

## 🧪 Tests Completados

### Test Suite 1: Lógica Unitaria (`test_conversation.py`)
- ✅ `is_follow_up()`: Detecta correctamente 0/1/2 búsquedas
- ✅ `IntentionDetector`: Clasifica 4 intenciones
- ✅ `EntityExtractor`: Extrae años, tipos, tópicos

### Test Suite 2: Integración (`test_integration.py`)
- ✅ Primera búsqueda registrada en sesión
- ✅ Seguimiento detecta unsatisfied → rama pidiendo detalles
- ✅ Unsatisfied CON detalles → re-búsqueda con query refinada
- ✅ Refinamiento → nueva búsqueda con tópicos
- ✅ Usuario satisfecho → termina conversación

### Test Suite 3: API End-to-End
- ✅ **TEST 1**: Nueva sesión "user_final" → Primera búsqueda "dictadura 1973"
  - Result: conversation_type=search, docs=5 ✅
  
- ✅ **TEST 2**: Misma sesión → Insatisfecho SIN detalles "No me sirven estos resultados"
  - Result: conversation_type=follow_up_branch, pide detalles ✅
  
- ✅ **TEST 3**: Misma sesión → Insatisfecho CON detalles "Quiero de 1975 a 1980"
  - Result: conversation_type=search, hace re-búsqueda, docs=5 ✅
  
- ✅ **TEST 4**: Satisfecho "Gracias, estos me sirven"
  - Result: conversation_type=follow_up_branch, mensaje de satisfacción ✅
  
- ✅ **TEST 5**: Nueva sesión "otro_usuario" → independencia confirmada ✅

---

## 📊 Modificaciones de Código

### Archivos Creados:
1. `chatbot/services/conversation.py` (191 líneas)
   - 4 clases + 2 funciones helper
   - 100% implementado y testeado

### Archivos Modificados:
1. `chatbot/api_chatbot.py`
   - Imports: Agregadas `List, Dict, Tuple, Optional` de typing
   - Imports conversation: `ConversationSession, IntentionDetector, EntityExtractor, DocumentComparator`
   - Global: `conversation_sessions = {}` (en-memory session storage)
   - Funciones added: `get_or_create_session()`, `handle_follow_up_message()`, `compare_and_format_results()`
   - Endpoint `/api/chat`: Lógica completa de multi-turno con ramificaciones

### Cambio Crítico:
- `ConversationSession.is_follow_up()`: Cambio de `> 1` a `>= 1`
  - Razón: Detectar seguimiento desde el SEGUNDO mensaje (cuando hay 1 búsqueda anterior)

---

## 🔄 Flujos de Usuario Implementados

### Flujo 1: Usuario Satisfecho
```
User: "dictadura 1973"
System: [5 documentos relevantes]

User: "Gracias, excelente"
System: "¡Excelente! ¿Hay algo más?"
```

### Flujo 2: Usuario Insatisfecho → Pedir Detalles
```
User: "dictadura 1973"
System: [5 documentos]

User: "No me sirven"
System: "¿Puedes ser más específico? ¿Qué tipo de documento? ¿Algún año?"
```

### Flujo 3: Usuario Insatisfecho → Re-búsqueda
```
User: "dictadura 1973"
System: [5 documentos]

User: "No me sirven"
System: "¿Puedes ser más específico?"

User: "Quiero de 1975 a 1980"
System: [Re-búsqueda con "dictadura 1973 Quiero de 1975 a 1980"]
```

### Flujo 4: Refinamiento
```
User: "dictadura"
System: [documentos]

User: "En realidad quiero derechos humanos 1980"
System: [Nueva búsqueda refinada]
```

---

## 🚀 Características Futuras

### Próximos Pasos (Cuando API Key esté disponible):
1. **Gemini-based Intent Detection**
   - Reemplazar regex por LLM para detección más sofisticada
   - `IntentionDetector.detect()` → call Gemini

2. **Embedding-based Document Similarity**
   - Reemplazar comparación por títulos con embeddings
   - `DocumentComparator` → use Gemini embeddings

3. **Session Persistence**
   - Cambiar `conversation_sessions = {}` (en-memory)
   - A Redis o database para multi-servidor

4. **User Authentication**
   - Session manager based on user login
   - Tied to user IDs instead of generic session_ids

5. **Advanced Ramification**
   - Context-aware responses from Gemini
   - Dynamic search strategy based on content analysis

---

## 🔐 Decisiones de Diseño

### 1. Session Storage (En-memory)
**Por qué**: MVP rápido, no requiere infra extra
**Alternativa futura**: Redis/Database para production

### 2. Regex-based Intent Detection
**Por qué**: Offline, rápido, determinista, no consume API
**Alternativa futura**: Gemini cuando API key esté funcionando

### 3. session_id en Request Body
**Por qué**: Compatible con frontend existente
**Mejor Pr ácticas**: Headers (X-Session-ID) o cookies para production

### 4. is_follow_up() >= 1
**Por qué**: Detectar ramificaciones desde el segundo mensaje
**Semántica**: "Si hay contexto previo, es un seguimiento"

---

## ✅ Checklist Completado

- [x] Crear `conversation.py` con 4 clases
- [x] Implementar `IntentionDetector` (regex + keywords)
- [x] Implementar `EntityExtractor` (años, tipos, tópicos)
- [x] Implementar `DocumentComparator` (similitud)
- [x] Importar en `api_chatbot.py`
- [x] Crear session storage dict
- [x] Implementar `get_or_create_session()`
- [x] Implementar `handle_follow_up_message()` con ramificaciones
- [x] Implementar `compare_and_format_results()`
- [x] Modificar endpoint `/api/chat` con lógica multi-turno
- [x] Fijar `is_follow_up()` para >= 1
- [x] Test 1: Primera búsqueda ✅
- [x] Test 2: Insatisfecho sin detalles ✅
- [x] Test 3: Insatisfecho con detalles ✅
- [x] Test 4: Satisfecho ✅
- [x] Test 5: Nueva sesión (independencia) ✅

---

## 📝 Notas de Desarrollo

**Línea de base**: El endpoint `/api/chat` antes era completamente stateless.

**Transformación**:
- ❌ Stateless → ✅ Stateful
- ❌ Una búsqueda por request → ✅ Contexto entre múltiples requests
- ❌ Sin intención detectada → ✅ Ramificación según intención
- ❌ Respuesta genérica → ✅ Respuesta adaptada al contexto

**Impacto UX**:
- Usuario puede ahora tener conversaciones naturales
- Sistema entiende satisfacción/insatisfacción
- Re-búsquedas inteligentes con contexto refinado
- Experiencia menos robótica, más conversacional

---

## 🎓 Lecciones Aprendidas

1. **Estado es crítico en conversaciones**
   - Sin sesiones, es imposible entender contexto
   - La ramificación requiere memoria

2. **Detección de intención es impactante**
   - Pequeñas palabras ("gracias", "en realidad") cambian el flujo
   - Regex sirve bien para MVP; Gemini para producción

3. **Entity extraction amplifica re-búsquedas**
   - Extraer años/tipos permite refinar automáticamente
   - Transforma "no me sirven" en búsqueda más específica

4. **Comparación de documentos es importante**
   - Usuarios notan si se repiten resultados
   - Marcar similares vs nuevos mejora UX

---

**Implementado y probado**: 5 de Diciembre de 2025  
**Estado Final**: ✅ PRODUCTION READY (MVP)  
**Próximo Paso**: Integración con frontend, upgrades cuando API key esté disponible
