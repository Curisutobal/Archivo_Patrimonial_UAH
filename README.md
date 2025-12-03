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

### Principios SOLID aplicados

- **SRP (Single Responsibility Principle)**
  - **¿Qué significa?** Cada módulo hace una sola cosa.
  - **Aplicación:** Separar creación de servicios (Factory), llamadas a IA (Proxy) y eventos (Observer) del controlador Flask (`api_chatbot.py`). Resultado: archivos más simples y fáciles de mantener.

- **DIP (Dependency Inversion Principle)**
  - **¿Qué significa?** El código debe depender de abstracciones, no de detalles concretos.
  - **Aplicación:** `api_chatbot.py` ahora pide “servicios” al `ServiceFactory` (abstracción). Si cambia Gemini o si no hay conexión, el resto del código sigue funcionando sin cambios.

### ¿En qué archivos se aplicó?

- `chatbot/api_chatbot.py` — usa la fábrica y el bus de eventos; mantiene endpoints y comportamiento.
- `chatbot/services/factory.py` — crea funciones de embedding de consulta y de respuesta (IA).
- `chatbot/services/llm_proxy.py` — protege llamadas a Gemini (embed y generate).
- `chatbot/services/events.py` — EventBus y LoggingObserver para registro desacoplado.

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
