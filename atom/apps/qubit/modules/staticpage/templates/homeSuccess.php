<?php decorate_with('layout_2col') ?>

<?php slot('title') ?>
  <h1><?php echo render_title($resource->getTitle(array('cultureFallback' => true))) ?></h1>
<?php end_slot() ?>

<?php slot('sidebar') ?>

  <?php echo get_component('menu', 'staticPagesMenu') ?>

  <section>
    <h2><?php echo __('Browse by') ?></h2>
    <ul>
      <?php $browseMenu = QubitMenu::getById(QubitMenu::BROWSE_ID) ?>
      <?php if ($browseMenu->hasChildren()): ?>
        <?php foreach ($browseMenu->getChildren() as $item): ?>
          <li><a href="<?php echo url_for($item->getPath(array('getUrl' => true, 'resolveAlias' => true))) ?>"><?php echo esc_specialchars($item->getLabel(array('cultureFallback' => true))) ?></a></li>
        <?php endforeach; ?>
      <?php endif; ?>
    </ul>
  </section>

  <?php echo get_component('default', 'popular', array('limit' => 10, 'sf_cache_key' => $sf_user->getCulture())) ?>

<?php end_slot() ?>

<div class="page">
  <div class="homepage-content">
    <h1>Archivo Patrimonial Universidad Alberto Hurtado</h1>
    <p>La plataforma Archivo Patrimonial pone a disposición del público en libre acceso, una selección de documentos con el objeto de acercar a los ciudadanos al patrimonio cultural y la memoria colectiva de nuestro país.</p>
    <p>A través de este portal, usted puede encontrar los siguientes fondos documentales:</p>

    <div class="fondos-list">
      <ul>
        <li><strong>Patricio Aylwin Azócar (1990-1994)</strong> - Documentos del primer presidente de la Concertación</li>
        <li><strong>Fotografías del Programa Padres e Hijos (1974-1976)</strong> - Imágenes del programa social durante la dictadura</li>
        <li><strong>Volantes Políticos (1973-1990)</strong> - Material de propaganda política de la época</li>
        <li><strong>Iglesias y Dictadura (1973-1990)</strong> - Documentos sobre el rol de la iglesia durante el régimen militar</li>
        <li><strong>Música Docta chilena</strong> - Colección de partituras y documentos musicales</li>
      </ul>
    </div>
  </div>
</div>

<?php if (QubitAcl::check($resource, 'update')): ?>
  <?php slot('after-content') ?>
    <section class="actions">
      <ul>
        <li><?php echo link_to(__('Edit'), array($resource, 'module' => 'staticpage', 'action' => 'edit'), array('title' => __('Edit this page'), 'class' => 'c-btn')) ?></li>
      </ul>
    </section>
  <?php end_slot() ?>
<?php endif; ?>
