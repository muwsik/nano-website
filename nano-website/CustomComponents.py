# -*- coding: cp1251 -*-

import streamlit.components.v1 as components

import io
import base64


def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format = "PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str 
   

def img_box(img1, img2, imgSize):
    html_part = f"""
        <div class="imgBox">
            <img src="data:image/png;base64,{pil_to_base64(img1)}" id="magnifiable-image1" class="default1" draggable="false" alt="">
            <img src="data:image/png;base64,{pil_to_base64(img2)}" id="magnifiable-image2" class="default2" draggable="false" alt="">
        </div>
    """

    css_part = f"""
        <style>
            .imgBox {{
                margin: 0;
                width: 1000px;
                height: 700px;
                overflow: hidden;
                display: flex;
                justify-content: center;
                align-items: center;
                position: relative;
            }}

            .imgBox img {{
                max-height: 80vh;
                width: auto;
                max-width: 100%;
                object-fit: contain;
                position: absolute;
            }}

            .default1, .default2 {{                
                width: 100%;
                height: 100%;
                object-fit: contain;
                background-size: cover;                
                user-select: none;
                pointer-events: none;
            }}

            .default1 {{
                z-index: 0;
            }}

            .default2 {{
                z-index: 1;
            }}

            .magnifier-lens {{
                position: absolute;
                border: 1px solid #000;
                border-radius: 2%;
                cursor: crosshair;
                box-shadow: 0 0 5px #000;
                pointer-events: none;
                visibility: hidden;
                background-repeat: no-repeat;
                z-index: 2;
            }}
        </style>
    """

    JS_part = """
        <script>
          var lastMouseX = 0;
          var lastMouseY = 0;
          var currentImageElement = null;
          var magnifierLens = document.createElement('div');
          magnifierLens.classList.add('magnifier-lens');
          document.body.appendChild(magnifierLens);
          var isLensEnabled = true;
          var magnificationFactor = 7;
          var magnifierSize = 200;
          var currentBgPosX = 0;
          var currentBgPosY = 0;

        //Переменная для смены фона для исследования другой области
          var imgChanged = false;
        //Присвоение переменным объектов изображений
          var imageElement = document.getElementById('magnifiable-image1');
          var imageElement2 = document.getElementById('magnifiable-image2');

          function setupMagnifier(imageElement) {
              magnifierLens.style.backgroundRepeat = 'no-repeat';
              magnifierLens.style.pointerEvents = 'none';
              magnifierLens.style.position = 'absolute';
              magnifierLens.style.border = '1px solid #000';
              magnifierLens.style.width = magnifierSize + 'px';
              magnifierLens.style.height = magnifierSize + 'px';
              magnifierLens.style.visibility = 'hidden';
              magnifierLens.style.boxShadow = '0 0 5px #909090';
              magnifierLens.style.cursor = 'crosshair';
            // Функция обновления размера линзы
              function updateMagnifierSize(deltaY) {
                  if (!currentImageElement) return;
          
                  var bounds = currentImageElement.getBoundingClientRect();
                  var mouseX = lastMouseX - bounds.left;
                  var mouseY = lastMouseY - bounds.top;

                  var oldSize = magnifierSize;
                  magnifierSize += deltaY > 0 ? -20 : 20;
                  magnifierSize = Math.max(100, Math.min(magnifierSize, 400));

                  magnificationFactor = 2 + (magnifierSize - 100) * (5 - 2) / (400 - 100);

                  magnifierLens.style.width = magnifierSize + 'px';
                  magnifierLens.style.height = magnifierSize + 'px';

                  updateLensPositionAndBackground();
              }
    
      
            //   Функция обновления позиции линзы и ее фона
              function updateLensPositionAndBackground() {
                  if (!currentImageElement) return;
          
                  var bounds = currentImageElement.getBoundingClientRect();
                  var mouseX = lastMouseX - bounds.left;
                  var mouseY = lastMouseY - bounds.top;
      
                  var lensX = mouseX - (magnifierSize / 2);
                  var lensY = mouseY - (magnifierSize / 2);
      
                  magnifierLens.style.left = (bounds.left + window.pageXOffset + lensX) + 'px';
                  magnifierLens.style.top = (bounds.top + window.pageYOffset + lensY) + 'px';
      
                  currentBgPosX = -((mouseX * magnificationFactor) - magnifierSize / 2);
                  currentBgPosY = -((mouseY * magnificationFactor) - magnifierSize / 2);
      
                  updateLensBackground();
              }
            //   Обработчик поведения мыши на первом изображении
              imageElement.addEventListener('mousemove', function(e) {
                  if (!isLensEnabled) return;
                  magnifierLens.style.visibility = 'visible';
                  lastMouseX = e.clientX;
                  lastMouseY = e.clientY;
                  currentImageElement = imageElement;
                  updateLensPositionAndBackground();
                  imageElement.style.cursor = 'none';
              });
            //   Обработчик поведения мыши на втором изображении
              imageElement2.addEventListener('mousemove', function(e) {
                  if (!isLensEnabled) return;
                  magnifierLens.style.visibility = 'visible';
                  lastMouseX = e.clientX;
                  lastMouseY = e.clientY;
                  currentImageElement = imageElement2;
                  updateLensPositionAndBackground();
                  imageElement2.style.cursor = 'none';
              });
            //  Обработчик колеса мыши для первого изображения
              imageElement.addEventListener('wheel', function(e) {
                  if (!isLensEnabled) return;
                  e.preventDefault();
                  updateMagnifierSize(e.deltaY);
              }, { passive: false });
             // Обработчик колеса мыши для второго изображения
              imageElement2.addEventListener('wheel', function(e) {
                  if (!isLensEnabled) return;
                  e.preventDefault();
                  updateMagnifierSize(e.deltaY);
              }, { passive: false });
            // Обработка сокрытия линзы первого изображения когда мышь покидает изображение
              imageElement.addEventListener('mouseleave', function() {
                  magnifierLens.style.visibility = 'hidden';
                  imageElement.style.cursor = 'default';
              });
              // Обработка сокрытия линзы второго изображения когда мышь покидает изображение
              imageElement2.addEventListener('mouseleave', function() {
                  magnifierLens.style.visibility = 'hidden';
                  imageElement2.style.cursor = 'default';
              });
              // Переменная для предотвращения вызова контекстного меню на изображениях
              let contextMenuHandler = function(e) {
                e.preventDefault();
              };
              imageElement.addEventListener('contextmenu', contextMenuHandler);
              imageElement2.addEventListener('contextmenu', contextMenuHandler);
          }
          //Функция быстрой смены области линзы
          function forceRefreshLens() {
              imgChanged = !imgChanged;
              console.log("imgChanged = " + imgChanged);
      
              // Если лупа видима, обновляем её сразу
              if (magnifierLens.style.visibility === 'visible' && currentImageElement) {
                  updateLensPositionAndBackground();
              }
          }
        //   Функция обновления фона линзы при смене области
          function updateLensBackground() {
                  var imgElement = imgChanged ? imageElement2 : imageElement;
                  magnifierLens.style.backgroundImage = `url('${imgElement.src}')`;
                  magnifierLens.style.backgroundSize = `${imgElement.width * magnificationFactor}px ${imgElement.height * magnificationFactor}px`;
                  magnifierLens.style.backgroundPosition = `${currentBgPosX}px ${currentBgPosY}px`;
          }
          setupMagnifier(imageElement);
          setupMagnifier(imageElement2);
        //   Обработчик нажатия ПКМ для смены области
          document.addEventListener('mousedown', function(e) {
              if (e.button === 2) {
                  e.preventDefault();
                  forceRefreshLens();
              }
          });
        // Обработчик смены изображений по нажатию ЛКМ
          let isFirstImageOnTop = true;
          imageElement.addEventListener('mousedown', function(e) {
            if (e.button === 0) {
                e.preventDefault();
                if (isFirstImageOnTop) {
                    imageElement.style.zIndex = 0;
                    imageElement2.style.zIndex = 1;
                    imgChanged = true;
                    console.log("imgChanged = " + imgChanged);
                } else {
                    imageElement.style.zIndex = 1;
                    imageElement2.style.zIndex = 0;
                    imgChanged = false;
                    console.log("imgChanged = " + imgChanged);
                }
                isFirstImageOnTop = !isFirstImageOnTop;
        
                // Если лупа видна, инициализируем ее обновление
                if (magnifierLens.style.visibility === 'visible') {
                    updateLensPositionAndBackground();
                }
            }
          });
        // Обработчик смены изображений по нажатию ЛКМ
          imageElement2.addEventListener('mousedown', function(e) {
              if (e.button === 0) {
                  e.preventDefault();
                  if (isFirstImageOnTop) {
                      imageElement.style.zIndex = 0;
                      imageElement2.style.zIndex = 1;
                      imgChanged = true;
                      console.log("imgChanged = " + imgChanged);
                  } else {
                      imageElement.style.zIndex = 1;
                      imageElement2.style.zIndex = 0;
                      imgChanged = false;
                      console.log("imgChanged = " + imgChanged);
                  }
                  isFirstImageOnTop = !isFirstImageOnTop;
          
                  // Если лупа видна, инициализируем ее обновление
                  if (magnifierLens.style.visibility === 'visible') {
                      updateLensPositionAndBackground();
                  }
              }
          });

          // Вынесенная функция для обновления позиции и фона линзы
          function updateLensPositionAndBackground() {
              if (!currentImageElement) return;
      
              var bounds = currentImageElement.getBoundingClientRect();
              var mouseX = lastMouseX - bounds.left;
              var mouseY = lastMouseY - bounds.top;
  
              var lensX = mouseX - (magnifierSize / 2);
              var lensY = mouseY - (magnifierSize / 2);
  
              magnifierLens.style.left = (bounds.left + window.pageXOffset + lensX) + 'px';
              magnifierLens.style.top = (bounds.top + window.pageYOffset + lensY) + 'px';
  
              currentBgPosX = -((mouseX * magnificationFactor) - magnifierSize / 2);
              currentBgPosY = -((mouseY * magnificationFactor) - magnifierSize / 2);
  
              updateLensBackground();
          }
        </script>
    """

    return components.html(
        html_part + css_part + JS_part,
        height = imgSize[1],
    )