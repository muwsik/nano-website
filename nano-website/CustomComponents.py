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
        <div id="imgBox">
            <img src="data:image/png;base64,{pil_to_base64(img1)}" id="magnifiable-image1" class="default1" draggable="false" alt="">
            <img src="data:image/png;base64,{pil_to_base64(img2)}" id="magnifiable-image2" class="default2" draggable="false" alt="">
        </div>
    """

    css_part = """
        <style>
            #imgBox{
                #resize: both;
                position: relative;
                margin: 5vw auto;
                margin-bottom: 1vw;
                width: 1000px;
                height: 800px;
                overflow: hidden;
            }

            .default1, .default2 {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-size: cover;
            }

            .default1{
                z-index: 0;
            }

            .default2{
                z-index: 1;
            }

            .magnifier-lens {
                position: absolute;
                border: 1px solid #000;
                cursor: crosshair;
                box-shadow: 0 0 5px #000;
                pointer-events: none;
                visibility: hidden;
                background-repeat: no-repeat;
                z-index: 2;
            }
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
            // Добавленные переменные для фикса съезжания:
            var magnifierSize = 200;
            var currentBgPosX = 0;
            var currentBgPosY = 0;

            function setupMagnifier(imageElement) {
                magnifierLens.style.backgroundRepeat = 'no-repeat';
                magnifierLens.style.pointerEvents = 'none';
                magnifierLens.style.position = 'absolute';
                magnifierLens.style.border = '1px solid #000';
                //magnifierLens.style.borderRadius = '50%';
                magnifierLens.style.width = magnifierSize + 'px';
                magnifierLens.style.height = magnifierSize + 'px';
                magnifierLens.style.visibility = 'hidden';
                magnifierLens.style.boxShadow = '0 0 5px #000';
                magnifierLens.style.cursor = 'crosshair';

                // Обновленная функция с фиксом позиционирования:
                function updateMagnifierSize(deltaY) {
                    if (!currentImageElement) return;

                    // Сохраним текущую позицию мыши относительно изображения
                    var bounds = currentImageElement.getBoundingClientRect();
                    var mouseX = lastMouseX - bounds.left;
                    var mouseY = lastMouseY - bounds.top;

                    // Сохраним текущую позицию центра линзы в координатах изображения
                    var currentCenterX = mouseX;
                    var currentCenterY = mouseY;

                    // Сохраняем текущее смещение фона относительно центра
                    var currentBgCenterX = -currentBgPosX + magnifierSize/2;
                    var currentBgCenterY = -currentBgPosY + magnifierSize/2;

                    // Обновили размер линзы
                    var oldSize = magnifierSize;
                    magnifierSize += deltaY > 0 ? -20 : 20;
                    magnifierSize = Math.max(100, Math.min(magnifierSize, 400));

                    // Обновляем коэффициент увеличения
                    var oldMagnification = magnificationFactor;
                    magnificationFactor = 2 + (magnifierSize - 100) * (5 - 2) / (400 - 100);

                    // Обновляем размеры линзы
                    magnifierLens.style.width = magnifierSize + 'px';
                    magnifierLens.style.height = magnifierSize + 'px';

                    // Пересчет позиции фона чтобы линза осталась на месте
                    var bgCenterX = currentCenterX * magnificationFactor;
                    var bgCenterY = currentCenterY * magnificationFactor;

                    currentBgPosX = -(bgCenterX - magnifierSize/2);
                    currentBgPosY = -(bgCenterY - magnifierSize/2);

                    // Обновляем позицию линзы
                    var lensX = mouseX - (magnifierSize / 2);
                    var lensY = mouseY - (magnifierSize / 2);

                    magnifierLens.style.left = (bounds.left + window.pageXOffset + lensX) + 'px';
                    magnifierLens.style.top = (bounds.top + window.pageYOffset + lensY) + 'px';

                    updateLensBackground(currentImageElement);
                }
    
                // Новая вспомогательная функция:
                function updateLensBackground(imgElement) {
                    magnifierLens.style.backgroundImage = `url('${imgElement.src}')`;
                    magnifierLens.style.backgroundSize = `${imgElement.width * magnificationFactor}px ${imgElement.height * magnificationFactor}px`;
                    magnifierLens.style.backgroundPosition = `${currentBgPosX}px ${currentBgPosY}px`;
                }

                function updateLensMagnification() {
                    if (!imageElement || !isLensEnabled) return;
      
                    var bounds = imageElement.getBoundingClientRect();
                    currentBgPosX = -((magnifierLens.offsetLeft - bounds.left) * magnificationFactor - magnifierSize / 2);
                    currentBgPosY = -((magnifierLens.offsetTop - bounds.top) * magnificationFactor - magnifierSize / 2);
      
                    updateLensBackground(imageElement);
                    magnifierLens.offsetHeight;
                }
      
                imageElement.addEventListener('mousemove', function(e) {
                    if (!isLensEnabled) return;
                    magnifierLens.style.visibility = 'visible';
                    lastMouseX = e.clientX;
                    lastMouseY = e.clientY;
          
                    var bounds = e.target.getBoundingClientRect();
                    var mouseX = e.clientX - bounds.left;
                    var mouseY = e.clientY - bounds.top;
      
                    var lensX = mouseX - (magnifierSize / 2);
                    var lensY = mouseY - (magnifierSize / 2);
      
                    magnifierLens.style.left = (bounds.left + window.pageXOffset + lensX) + 'px';
                    magnifierLens.style.top = (bounds.top + window.pageYOffset + lensY) + 'px';
      
                    currentBgPosX = -((mouseX * magnificationFactor) - magnifierSize / 2);
                    currentBgPosY = -((mouseY * magnificationFactor) - magnifierSize / 2);
      
                    updateLensBackground(imageElement);
                    imageElement.style.cursor = 'none';
                    currentImageElement = imageElement;
                });

      
                imageElement.addEventListener('wheel', function(e) {
                    if (!isLensEnabled) return;
                    e.preventDefault();
                    updateMagnifierSize(e.deltaY);
                }, { passive: false });
    
                imageElement.addEventListener('mouseleave', function() {
                    magnifierLens.style.visibility = 'hidden';
                    imageElement.style.cursor = 'default';
                });
                let contextMenuHandler = function(e) {
                e.preventDefault();
                //alert("Контекстное меню заблокировано!");
                };
                imageElement.addEventListener('contextmenu', contextMenuHandler);
            }
            function updateLens() {
            if (!isLensEnabled || !currentImageElement) return;
    
            // Переключаем изображение
            if (currentImageElement === imageElement) {
                currentImageElement = imageElement2;
            } else {
                currentImageElement = imageElement;
            }
    
            const bounds = currentImageElement.getBoundingClientRect();
            // Используем последние известные координаты мыши
            const mouseX = lastMouseX - bounds.left;
            const mouseY = lastMouseY - bounds.top;
    
            // Вычисляем новую позицию линзы
            const lensX = mouseX - (magnifierSize / 2);
            const lensY = mouseY - (magnifierSize / 2);
    
            // Обновляем позицию линзы
            magnifierLens.style.left = (bounds.left + window.pageXOffset + lensX) + 'px';
            magnifierLens.style.top = (bounds.top + window.pageYOffset + lensY) + 'px';
    
            // Обновляем фон
            currentBgPosX = -((mouseX * magnificationFactor) - magnifierSize / 2);
            currentBgPosY = -((mouseY * magnificationFactor) - magnifierSize / 2);
    
            magnifierLens.style.backgroundImage = `url('${currentImageElement.src}')`;
            magnifierLens.style.backgroundSize = `${currentImageElement.width * magnificationFactor}px ${currentImageElement.height * magnificationFactor}px`;
            magnifierLens.style.backgroundPosition = `${currentBgPosX}px ${currentBgPosY}px`;
            }
            function forceRefreshLens() {
            updateLens();
            }
            var imageElement = document.getElementById('magnifiable-image1');
            var imageElement2 = document.getElementById('magnifiable-image2');
            setupMagnifier(imageElement);
            setupMagnifier(imageElement2);
            document.addEventListener('mousedown', function(e) {
                if (e.button === 2) {
                    e.preventDefault();
                    forceRefreshLens();
                }
            });

            let isFirstImageOnTop = true;
            document.addEventListener('mousedown', function(e) {
                if (e.button === 1) {
                    if (isFirstImageOnTop) {
                        imageElement.style.zIndex = 0;
                        imageElement2.style.zIndex = 1;
                    } else {
                        imageElement.style.zIndex = 1;
                        imageElement2.style.zIndex = 0;
                    }
                    isFirstImageOnTop = !isFirstImageOnTop;
                }
            });

            var imageElement = document.getElementById('magnifiable-image1');
            var imageElement2 = document.getElementById('magnifiable-image2');
            setupMagnifier(imageElement);
            setupMagnifier(imageElement2);
        </script>
    """

    return components.html(
        html_part + css_part + JS_part,
        height = imgSize[0],
    )