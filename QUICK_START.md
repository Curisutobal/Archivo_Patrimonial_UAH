# 🚀 Guía Rápida: Ejecutar el Proyecto (Para Novatos)

Si eres nuevo en programación y acabas de descargar este proyecto, sigue **exactamente** estos comandos en orden. No necesitas entender nada de lo que haya adentro. Solo copia y pega.

---

## Paso 1: Abre una terminal

**En Windows:**
- Opción A: Click derecho en la carpeta del proyecto → "Abrir en Terminal"
- Opción B: Abre **PowerShell** o **CMD**

**En Mac/Linux:**
- Abre **Terminal**

---

## Paso 2: Colócate en la carpeta del proyecto

Si ya no estás adentro, escribe esto:

```bash
cd ruta/a/tu/carpeta/Archivo_Patrimonial-main
```

(Reemplaza `ruta/a/tu/carpeta` con donde descargaste el proyecto)

---

## Paso 3: Inicia Docker

Escribe este comando y espera a que termine (puede tardar 2-5 minutos la primera vez):

```bash
docker compose up
```

**¿Qué estoy haciendo?** Docker está bajando e iniciando toda la aplicación: base de datos, buscador, servidor web, y chatbot.

**Espera a que veas esto en la terminal:**
```
chatbot_api  | * Running on http://0.0.0.0:5000
```

Eso significa que está listo. ✅

---

## Paso 4: Abre la aplicación en tu navegador

En otra ventana del navegador (Chrome, Firefox, Edge, Safari), ve a:

```
http://localhost:8080
```

**¿Qué debería ver?** Una página web con el archivo patrimonial y un chatbot en la esquina (o donde esté configurado).

---

## Paso 5: ¡Usa el chatbot!

Escribe en el chat:
- "violaciones derechos humanos"
- "fotografías 1973"
- "dictadura militar"

El chatbot te mostrará documentos relacionados.

---

## Si algo no funciona...

### Error: "No se encuentra Docker"
- **Solución:** Instala Docker Desktop desde https://www.docker.com/products/docker-desktop

### Error: "Puerto 8080 ya está en uso"
- **Solución:** Cierra otras aplicaciones que usen ese puerto, o abre la carpeta del proyecto en VS Code y edita `docker-compose.yml` (línea ~70, cambia `8080:80` a `8082:80`)

### El navegador dice "No se puede conectar"
- **Solución:** Espera 30 segundos más. Docker a veces tarda en iniciar todo. Actualiza la página (Ctrl+R o Cmd+R)

### Terminal dice "error: volumes"
- **Solución:** Asegúrate de estar en la carpeta correcta (la que tiene `docker-compose.yml`). Luego escribe:
  ```bash
  docker compose down
  docker compose up
  ```

---

## Cuando termines: Detén Docker

En la terminal donde está corriendo Docker, presiona:

```
Ctrl + C
```

Eso es todo. ✅

---

## Lo que está pasando "detrás de camarines" (opcional)

Si tienes curiosidad:

- **Base de datos:** Guarda todos los documentos del archivo
- **Buscador:** Encuentra rápido los documentos que necesitas
- **Servidor web:** Sirve la página que ves en el navegador
- **Chatbot:** Responde tus preguntas en tiempo real

Pero **no necesitas saber esto para usar la aplicación.** Solo funcionan.

---

## ¿Preguntas?

Si algo no está claro o necesitas ayuda:
1. Lee nuevamente este archivo de arriba a abajo
2. Verifica que Docker esté instalado
3. Asegúrate de estar en la carpeta correcta (la que tiene `docker-compose.yml`)

¡Eso es todo! 🎉
