# Archivo Patrimonial UAH — Guía rápida y notas de diseño

Esta guía explica cómo probar la API del chatbot y resume los cambios de diseño realizados (patrones de diseño y principios SOLID) con un lenguaje claro y directo.

## Probar la API

- Backend expuesto vía Nginx en `http://localhost:8080`.
- Endpoints principales:
  - `GET /api/health` — estado y métricas.
  - `POST /api/chat` — recibe la consulta del usuario.

### PowerShell (Windows) — forma simple (formulario)

Usa cuerpo de formulario para evitar problemas de codificación de JSON en PowerShell 5.1:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://localhost:8080/api/chat' -ContentType 'application/x-www-form-urlencoded' -Body 'query=hola'
Invoke-RestMethod -Method Post -Uri 'http://localhost:8080/api/chat' -ContentType 'application/x-www-form-urlencoded' -Body 'query=dictadura militar'
```

### PowerShell (Windows) — forma JSON correcta (UTF-8)

Si prefieres JSON, envía el cuerpo como bytes UTF-8 con cabeceras explícitas:

```powershell
$json = '{"query":"fotografias 1975"}';
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json);
Invoke-RestMethod -Method Post -Uri 'http://localhost:8080/api/chat' -Headers @{ 'Content-Type'='application/json; charset=utf-8'; 'Accept'='application/json' } -Body $bytes
```

### curl (Git Bash/WSL/Mac/Linux)

```bash
curl -s -X POST 'http://localhost:8080/api/chat' -H 'Content-Type: application/json' -d '{"query":"fotografias 1975"}'
```

## ¿Qué cambió del código?

**Actualización reciente:** Se mejoró el sistema de búsqueda con:

### Mejora del nivel de respuesta (Sistema de sugerencias inteligentes)

El chatbot ahora analiza **todas las búsquedas** de forma automática y ofrece ayuda contextual cuando los resultados pueden no ser suficientes:

1. **Detección automática de consultas genéricas**
   - Si escribes términos muy amplios (ej: "dictadura", "fotografías", "gobierno", "historia"), el sistema lo detecta automáticamente
   - Te sugiere refinamientos específicos: añadir años, contexto, o términos relacionados
   - **Funciona con cualquier búsqueda**, no solo con palabras específicas

2. **Análisis de documentos encontrados**
   - Extrae automáticamente **temas comunes** de los títulos de resultados
   - Detecta **años mencionados** (1973, 1974, 1980, etc.)
   - Identifica **palabras clave frecuentes** que puedes usar para refinar

3. **Sugerencias contextuales personalizadas**
   - Si buscas "dictadura" → sugiere: "dictadura años 70", "dictadura 1973", "dictadura documentos"
   - Si buscas "derechos humanos 1980" → sugiere temas encontrados: "solicita", "casos", "violaciones"
   - Si buscas "MIR" → sugiere años o contextos detectados en los resultados
   - **Las sugerencias cambian según tu consulta y los documentos encontrados**

4. **Búsqueda por keywords (respaldo automático)**
   - Si la API de Gemini no está disponible, el sistema usa búsqueda por coincidencia de palabras en títulos
   - Funciona con cualquier término sin necesidad de embeddings
   - Calcula relevancia por número de palabras coincidentes

**Objetivo:** Si los 6 documentos sugeridos no son exactamente lo que buscabas, el chatbot te ayuda a refinar automáticamente sin necesidad de adivinar qué más buscar.

---

Se reforzó el backend del chatbot (`chatbot/api_chatbot.py`) con tres patrones de diseño y dos principios SOLID. El objetivo: mejorar orden, seguridad y mantenibilidad sin cambiar el comportamiento.

### Patrones de diseño utilizados

- **Abstract Factory** — `chatbot/services/factory.py`
  - **¿Qué hace?** Centraliza cómo se crean las funciones de “embedding” (para búsqueda) y la de “respuesta” (IA), dependiendo si la API de Gemini está disponible o no.
  - **¿Por qué aquí?** Permite cambiar la estrategia (usar Gemini o un reemplazo básico) sin tocar el resto del código. Esto reduce el acoplamiento y hace el sistema más flexible.

- **Proxy** — `chatbot/services/llm_proxy.py`
  - **¿Qué hace?** Envuelve las llamadas a Gemini para manejarlas con seguridad (errores, indisponibilidad) y devolver valores controlados en caso de fallo.
  - **¿Por qué aquí?** Evita que errores externos (API) rompan el flujo del servidor. El Proxy es perfecto para poner “una capa de seguridad” sin reescribir la lógica de negocio.

- **Observer** — `chatbot/services/events.py`
  - **¿Qué hace?** Implementa un bus de eventos simple (publicar/suscribir) y un observador de logging (`LoggingObserver`).
  - **¿Por qué aquí?** Permite registrar lo que ocurre (recibir consultas, tipo detectado, búsqueda hecha, respuesta generada) sin mezclar logs con la lógica central. Así podemos añadir métricas o auditoría sin tocar el flujo principal.

- **Strategy (Explícito)** — `chatbot/services/conversation.py`
  - **¿Qué hace?** Define abstracciones base (`IntentionStrategy`, `EntityStrategy`, `SimilarityStrategy`) que permiten múltiples implementaciones intercambiables.
  - **¿Por qué aquí?** El chatbot necesita ser extensible: hoy usamos regex para detección, mañana queremos Gemini o ML. Las estrategias permitenSwitch sin tocar el código existente.

### Principios SOLID aplicados

- **SRP (Single Responsibility Principle)**
  - **¿Qué significa?** Cada módulo hace una sola cosa.
  - **Aplicación:** Separar creación de servicios (Factory), llamadas a IA (Proxy) y eventos (Observer) del controlador Flask (`api_chatbot.py`). Resultado: archivos más simples y fáciles de mantener.
  - **En conversation.py:** Cada clase (`ConversationSession`, `IntentionDetector`, `EntityExtractor`, `DocumentComparator`) tiene UNA responsabilidad específica. Ninguna mezcla lógicas.

- **OCP (Open/Closed Principle) — Mejorado en conversation.py**
  - **¿Qué significa?** Las clases deben ser abiertas para EXTENSIÓN, cerradas para MODIFICACIÓN.
  - **Aplicación en conversation.py:**
    - Abstracciones base: `IntentionStrategy`, `EntityStrategy`, `SimilarityStrategy`
    - Nuevas implementaciones heredan sin tocar código existente
    - **Ejemplos de extensión futura:**
      ```python
      class GeminiIntentionDetector(IntentionStrategy):
          """Detección mejorada con IA (sin modificar código actual)"""
          def detect(self, message):
              # Usa Gemini en lugar de regex
              return genai.detect_intention(message)
      
      class EmbeddingComparator(SimilarityStrategy):
          """Similitud con embeddings (sin modificar código actual)"""
          def calculate_topic_similarity(self, docs1, docs2):
              # Usa embeddings en lugar de palabras
              return embedding_based_similarity(docs1, docs2)
      ```
    - El resto del código sigue funcionando sin cambios

- **DIP (Dependency Inversion Principle) — Mejorado en conversation.py**
  - **¿Qué significa?** Depender de abstracciones, no de implementaciones concretas.
  - **Aplicación en conversation.py:**
    - Inyección de dependencias en constructores
    - Patrones personalizables sin modificar la clase
    - **Ejemplo:**
      ```python
      # Uso por defecto (regex)
      detector = IntentionDetector()
      
      # Uso personalizado (patrones custom)
      custom_patterns = {
          'satisfied': [r'...'],
          'unsatisfied': [r'...']
      }
      detector = IntentionDetector(patterns=custom_patterns)
      
      # Uso futuro (estrategia diferente completamente)
      detector = GeminiIntentionDetector()  # Otro proveedor, mismo interfaz
      ```
    - En `api_chatbot.py`: instancias de estrategias inyectadas como globales
      ```python
      intention_detector = IntentionDetector()      # Intercambiable
      entity_extractor = EntityExtractorImpl()       # Intercambiable
      document_comparator = DocumentComparator()    # Intercambiable
      ```

- **DIP (Dependency Inversion Principle) — Clásico**
  - **¿Qué significa?** El código debe depender de abstracciones, no de detalles concretos.
  - **Aplicación:** `api_chatbot.py` ahora pide "servicios" al `ServiceFactory` (abstracción). Si cambia Gemini o si no hay conexión, el resto del código sigue funcionando sin cambios.

### ¿En qué archivos se aplicó?

- `chatbot/api_chatbot.py` — usa la fábrica, bus de eventos y estrategias inyectadas; mantiene endpoints y comportamiento.
- `chatbot/services/factory.py` — crea funciones de embedding y respuesta (IA).
- `chatbot/services/llm_proxy.py` — protege llamadas a Gemini.
- `chatbot/services/events.py` — EventBus y LoggingObserver para registro desacoplado.
- **`chatbot/services/conversation.py`** — ⭐ **NUEVO**: Gestión multi-turno, estrategias base (OCP), inyección (DIP)
  - `ConversationSession` — gestiona historial por usuario (SRP)
  - `IntentionStrategy` (abstracta) / `IntentionDetector` (regex) — detecta intención (OCP+DIP)
  - `EntityStrategy` (abstracta) / `EntityExtractorImpl` (regex) — extrae entidades (OCP+DIP)
  - `SimilarityStrategy` (abstracta) / `DocumentComparator` — compara documentos (OCP+DIP)

## ⭐ Conversación Multi-turno (Nuevo)

El chatbot ahora mantiene contexto entre múltiples mensajes y adapta su lógica según la intención del usuario:

### Flujos de conversación

1. **Usuario satisfecho**
   ```
   User:  "dictadura 1973"
   Bot:   [5 documentos relevantes]
   
   User:  "Gracias, perfecto"
   Bot:   "¡Excelente! ¿Hay algo más que quieras explorar?"
   ```

2. **Usuario insatisfecho → pide detalles**
   ```
   User:  "derechos humanos"
   Bot:   [6 documentos]
   
   User:  "No encuentro lo que buscaba"
   Bot:   "¿Puedes ser más específico? ¿Años? ¿Tipo de documento? ¿Tema?"
   ```

3. **Usuario insatisfecho + proporciona detalles → re-búsqueda**
   ```
   User:  "No encuentro"
   Bot:   "¿Puedes ser más específico?"
   
   User:  "Quiero de 1975 a 1980"
   Bot:   [Nueva búsqueda refinada con años]
   ```

4. **Refinamiento (cambio de tema)**
   ```
   User:  "dictadura"
   Bot:   [documentos sobre dictadura]
   
   User:  "En realidad quiero derechos humanos 1980"
   Bot:   [Nueva búsqueda adaptada]
   ```

### Cómo funciona (tecnicamente)

- **`session_id`** en cada request identifica al usuario y mantiene historial
- **`IntentionDetector`** clasifica el mensaje: satisfied / unsatisfied / refinement
- **`EntityExtractor`** obtiene contexto: años, tipos de doc, tópicos
- **`DocumentComparator`** marca documentos como repetidos (🔄) o nuevos (✨)
- **Ramificación inteligente:** El endpoint `/api/chat` cambia comportamiento según intención

### Extensibilidad

Todas las estrategias son intercambiables sin modificar el código:

```python
# Hoy: regex (rápido, local)
detector = IntentionDetector()

# Mañana: Gemini (más sofisticado)
detector = GeminiIntentionDetector()

# El resto del código sigue igual (polimorfismo)
intention = detector.detect(message)  # Funciona con ambos
```

## ¿Por qué no usamos otros patrones (y cuáles)?

- **Singleton:** Evitado para no introducir estados globales difíciles de testear. La configuración ya se maneja claramente con variables de entorno (p. ej., `GEMINI_API_KEY`).
- **Decorator:** Útil para añadir comportamiento dinámico, pero el objetivo aquí era separar responsabilidades y proteger llamadas externas; el Proxy satisface mejor esa necesidad.
- **Strategy “pura”:** La fábrica ya selecciona estrategias (con o sin GENAI). Usar Strategy adicional habría duplicado estructuras sin aportar claridad.
- **Facade:** Nginx y Flask ya sirven como “fachada” de entrada. Añadir otra fachada no resolvía un problema concreto.

## Seguridad y configuración

- La clave de Gemini ahora se lee desde `.env` y `docker-compose.yml` (variable `GEMINI_API_KEY`).
- **Importante:** Si ves errores `403 Your API key was reported as leaked`, necesitas generar una nueva clave en [Google AI Studio](https://aistudio.google.com/app/apikey) y actualizar tu `.env`.
- El sistema funciona en modo degradado (búsqueda por keywords) si Gemini no está disponible.
- Para evitar exponer secretos o binarios grandes, `.gitignore` incluye:
  - `.env`, `chatbot/.env`
  - `*.pkl`, `chatbot/embeddings_cache.pkl`
  - `atom/vendor/`, `atom/cache/`, `atom/log/`

### Configuración inicial después de clonar el repositorio

Si clonas este proyecto desde GitHub, necesitarás recrear algunos archivos que no se suben por seguridad o tamaño:

1. **Crear archivo `.env` en la raíz del proyecto:**
   ```bash
   GEMINI_API_KEY=tu_clave_aqui
   ```
   Obtén tu clave en [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Instalar dependencias PHP de AtoM (opcional, solo si usas AtoM):**
   ```bash
   cd atom
   composer install
   ```

3. **Iniciar los contenedores Docker:**
   ```bash
   docker compose up -d
   ```

4. **El sistema generará automáticamente:**
   - `chatbot/embeddings_cache.pkl` — se crea en el primer arranque si GENAI está disponible
   - `atom/cache/` — cache de Symfony (se regenera automáticamente)

## Estado y salud

- `GET /api/health` devuelve el estado (documentos cargados, embeddings disponibles y si la IA está activa).
- Si la IA no está disponible, el sistema sigue funcionando: muestra documentos relevantes y enlaces sin detener el servicio.

## Preguntas frecuentes

- **"No veo resultados de búsqueda"**: El sistema ahora usa búsqueda por palabras clave como respaldo. Si no aparece nada, reformula con términos más específicos (ej.: "derechos humanos años 80", "MIR", "fotografías 1975").
- **"Mi POST JSON falla en PowerShell"**: Usa el método de formulario o el envío de bytes UTF-8 con cabeceras (ver arriba).
- **"Veo sugerencias debajo de los resultados"**: Esto es nuevo. El chatbot analiza los documentos encontrados y te sugiere cómo refinar la búsqueda si es muy amplia.
- **"¿Por qué dice 'Tu búsqueda es amplia'?"**: Consultas como "dictadura", "gobierno", "fotografías" solas son muy genéricas. El sistema te pide que añadas más contexto (años, temas específicos, etc.).

---

Si quieres, puedo añadir ejemplos de métricas con el `EventBus` (tiempos de respuesta) o una pequeña batería de pruebas para el servicio de búsqueda.
