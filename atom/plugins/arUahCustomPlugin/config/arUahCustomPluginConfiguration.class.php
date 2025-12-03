<?php

class arUahCustomPluginConfiguration extends sfPluginConfiguration
{
  public function initialize()
  {
    $this->dispatcher->connect('context.load_factories', array($this, 'listenToContextLoadFactoriesEvent'));
  }

  public function listenToContextLoadFactoriesEvent(sfEvent $event)
  {
    $context = $event->getSubject();
    $context->response->addStylesheet('/plugins/arUahCustomPlugin/css/main.css', 'last');
  }
}
