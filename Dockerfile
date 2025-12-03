FROM php:7.4-apache-bullseye

# Aumentar límite de memoria de PHP (evita errores de Allowed memory size)
RUN echo "memory_limit=1024M" > /usr/local/etc/php/conf.d/memory-limit.ini

# Instalar dependencias del sistema y herramientas de compilación
RUN apt-get update && apt-get install -y \
    git unzip wget curl gnupg2 \
    libicu-dev g++ libpng-dev libjpeg-dev libfreetype6-dev \
    libxslt1-dev zlib1g-dev libzip-dev \
    imagemagick ghostscript poppler-utils openjdk-11-jre-headless \
    build-essential autoconf pkg-config re2c \
 && rm -rf /var/lib/apt/lists/*

# --- Extensiones PHP (sin mbstring por ahora) ---
RUN docker-php-ext-install intl
RUN docker-php-ext-install pdo pdo_mysql
RUN docker-php-ext-configure gd --with-freetype --with-jpeg \
 && docker-php-ext-install gd
RUN docker-php-ext-install xsl
RUN docker-php-ext-install zip
RUN docker-php-ext-install opcache

# APCu desde PECL
RUN pecl install apcu \
 && docker-php-ext-enable apcu

# --- Composer ---
RUN curl -sS https://getcomposer.org/installer | php \
    -- --install-dir=/usr/local/bin --filename=composer

# --- Descargar AtoM ---
RUN git clone -b stable/2.7.x https://github.com/artefactual/atom.git /var/www/html

# Permisos
RUN chown -R www-data:www-data /var/www/html

# Activar mod_rewrite
RUN a2enmod rewrite

# Configurar Apache para permitir .htaccess
RUN sed -i '/<Directory \/var\/www\/>/,/<\/Directory>/ s/AllowOverride None/AllowOverride All/' /etc/apache2/apache2.conf

# Copiar .htaccess al directorio de AtoM
COPY .htaccess /var/www/html/.htaccess

# Asegurar permisos correctos para .htaccess
RUN chown www-data:www-data /var/www/html/.htaccess
