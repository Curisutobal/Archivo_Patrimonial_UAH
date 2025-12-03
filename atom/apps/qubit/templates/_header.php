<?php echo get_component('default', 'updateCheck') ?>

<?php echo get_component('default', 'privacyMessage') ?>

<?php if ($sf_user->isAdministrator() && (string)QubitSetting::getByName('siteBaseUrl') === ''): ?>
  <div class="site-warning">
    <?php echo link_to(__('Please configure your site base URL'), 'settings/siteInformation', array('rel' => 'home', 'title' => __('Home'))) ?>
  </div>
<?php endif; ?>

<header id="top-bar">

  <?php if (sfConfig::get('app_toggleLogo')): ?>
    <?php echo link_to(image_tag('/plugins/arUahCustomPlugin/images/logo.png', array('alt' => 'Archivo Patrimonial UAH', 'style' => 'height: 60px;')), '@homepage', array('id' => 'logo', 'rel' => 'home')) ?>
  <?php endif; ?>

  <?php if (sfConfig::get('app_toggleTitle')): ?>
    <h1 id="site-name">
      <?php echo link_to('<span>'.esc_specialchars(sfConfig::get('app_siteTitle')).'</span>', '@homepage', array('rel' => 'home', 'title' => __('Home'))) ?>
    </h1>
  <?php endif; ?>

  <nav>

    <?php echo get_component('menu', 'userMenu') ?>

    <?php echo get_component('menu', 'quickLinksMenu') ?>

    <?php if (sfConfig::get('app_toggleLanguageMenu')): ?>
      <?php echo get_component('menu', 'changeLanguageMenu') ?>
    <?php endif; ?>

    <?php echo get_component('menu', 'clipboardMenu') ?>

    <?php echo get_component('menu', 'mainMenu', array('sf_cache_key' => $sf_user->getCulture().$sf_user->getUserID())) ?>

  </nav>

  <div id="search-bar">

    <?php echo get_component('menu', 'browseMenu', array('sf_cache_key' => $sf_user->getCulture().$sf_user->getUserID())) ?>

    <?php echo get_component('search', 'box') ?>

  </div>

  <?php echo get_component_slot('header') ?>

</header>

  <?php if (sfConfig::get('app_toggleDescription')): ?>
  <div id="site-slogan">
    <div class="container">
      <div class="row">
        <div class="span12">
          <span><?php echo esc_specialchars(sfConfig::get('app_siteDescription')) ?></span>
        </div>
      </div>
    </div>
  </div>
<?php endif; ?>

<!-- Banner informativo UAH -->
<div class="bg-secondary text-white py-2">
  <div class="container">
    <div class="row">
      <div class="col-12 text-center">
        <strong>Archivo Patrimonial</strong> - Universidad Alberto Hurtado
      </div>
    </div>
  </div>
</div>

<!-- Chatbot Integration -->
<div id="chatbot-container" style="display: none; position: fixed; bottom: 20px; right: 20px; width: 350px; height: 500px; background: white; border: 1px solid #ccc; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); z-index: 10000; font-family: Arial, sans-serif;">
    <div id="chatbot-header" style="background: #007cba; color: white; padding: 10px; border-radius: 10px 10px 0 0; cursor: move;">
        <span>🤖 Chatbot Archivo UAH</span>
        <button id="chatbot-close" style="float: right; background: none; border: none; color: white; font-size: 20px; cursor: pointer;">×</button>
    </div>
    <div id="chatbot-messages" style="height: 400px; overflow-y: auto; padding: 10px; background: #f9f9f9;">
        <div style="text-align: center; color: #666; margin-top: 180px;">
            ¡Hola! Soy el chatbot del Archivo Patrimonial UAH.<br>
            ¿En qué puedo ayudarte?
        </div>
    </div>
    <div id="chatbot-input" style="padding: 10px; border-top: 1px solid #ccc;">
        <input type="text" id="chatbot-user-input" placeholder="Escribe tu pregunta..." style="width: 70%; padding: 5px; border: 1px solid #ccc; border-radius: 3px;">
        <button id="chatbot-send" style="width: 25%; padding: 5px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer;">Enviar</button>
    </div>
</div>

<div id="chatbot-icon" style="position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; background: #007cba; border-radius: 50%; cursor: pointer; z-index: 9999; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">
    <span style="color: white; font-size: 24px;">🤖</span>
</div>

<script>
// Chatbot functionality - Versión Corregida para AtoM
document.addEventListener('DOMContentLoaded', function() {
    const chatbotIcon = document.getElementById('chatbot-icon');
    const chatbotContainer = document.getElementById('chatbot-container');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotInput = document.getElementById('chatbot-user-input');
    const chatbotMessages = document.getElementById('chatbot-messages');

    // Toggle chatbot visibility
    chatbotIcon.addEventListener('click', function() {
        chatbotContainer.style.display = chatbotContainer.style.display === 'none' ? 'block' : 'none';
        chatbotIcon.style.display = chatbotContainer.style.display === 'none' ? 'flex' : 'none';
        // Auto-focus al abrir
        if (chatbotContainer.style.display === 'block') {
            chatbotInput.focus();
        }
    });

    // Close chatbot
    chatbotClose.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
        chatbotIcon.style.display = 'flex';
    });

    // Send message function
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user');
        chatbotInput.value = '';
        chatbotInput.disabled = true; // Evitar doble envío
        chatbotSend.disabled = true;

        // Mostrar indicador de carga
        const loadingDiv = addMessage('Escribiendo...', 'bot-loading');

        // Send to API (Apuntando al puerto 8080 donde está Nginx)
        fetch('http://localhost:8080/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            // CORRECCIÓN 1: Cambiamos 'message' por 'query' para que Python lo entienda
            body: JSON.stringify({ query: message })
        })
        .then(response => {
            if (!response.ok) throw new Error('Error en la red');
            return response.json();
        })
        .then(data => {
            // Eliminar mensaje de carga
            if (loadingDiv) loadingDiv.remove();

            if (data.response) {
                addMessage(data.response, 'bot');
            } else {
                addMessage('No entendí eso, ¿puedes repetirlo?', 'bot');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            if (loadingDiv) loadingDiv.remove();
            addMessage('⚠️ Error de conexión. Asegúrate de que el sistema esté encendido (Puerto 8080).', 'bot');
        })
        .finally(() => {
            chatbotInput.disabled = false;
            chatbotSend.disabled = false;
            chatbotInput.focus();
        });
    }

    // Add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            margin: 5px 0;
            padding: 8px 12px;
            border-radius: 8px;
            max-width: 85%;
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.4;
        `;

        if (sender === 'user') {
            messageDiv.style.cssText += 'background: #007cba; color: white; margin-left: auto; text-align: right;';
            messageDiv.textContent = text; // Usuario siempre es texto plano
        } else if (sender === 'bot-loading') {
            messageDiv.style.cssText += 'color: #666; font-style: italic; font-size: 12px;';
            messageDiv.textContent = text;
        } else {
            // Bot: Estilo gris claro
            messageDiv.style.cssText += 'background: white; color: #333; border: 1px solid #ddd; margin-right: auto;';
            
            // CORRECCIÓN 2: Usar innerHTML para que se vean los enlaces y negritas
            // Y asegurar que los enlaces se abran en nueva pestaña
            messageDiv.innerHTML = text.replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" style="color: #007cba; text-decoration: underline;" ');
        }

        chatbotMessages.appendChild(messageDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        
        return messageDiv;
    }

    // Send on button click
    chatbotSend.addEventListener('click', sendMessage);

    // Send on Enter key
    chatbotInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Make chatbot draggable
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };

    const header = document.getElementById('chatbot-header');
    
    header.addEventListener('mousedown', function(e) {
        // Evitar arrastrar si se hace clic en el botón cerrar
        if (e.target.id === 'chatbot-close') return;
        
        isDragging = true;
        dragOffset.x = e.clientX - chatbotContainer.offsetLeft;
        dragOffset.y = e.clientY - chatbotContainer.offsetTop;
    });

    document.addEventListener('mousemove', function(e) {
        if (isDragging) {
            e.preventDefault();
            chatbotContainer.style.left = (e.clientX - dragOffset.x) + 'px';
            chatbotContainer.style.top = (e.clientY - dragOffset.y) + 'px';
            chatbotContainer.style.right = 'auto'; 
            chatbotContainer.style.bottom = 'auto';
        }
    });

    document.addEventListener('mouseup', function() {
        isDragging = false;
    });
});
</script>