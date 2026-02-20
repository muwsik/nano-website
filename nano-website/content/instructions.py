import streamlit as st
import numpy as np
import datetime

def Header():
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
   
    
def About():
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
            <br>It will help you to detect nanoparticles in the image and calculate their statictics.
        </div>
    """, unsafe_allow_html = True)

    st.markdown("""
        <div class = 'about' style = "padding-bottom: 25px;">
            Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
        </div>
    """, unsafe_allow_html = True)


def DetectResult(countNP, time):
    st.markdown(f"""
        <p class = 'text'>
            Nanoparticles detected: <b>{countNP}</b> ({time//60}m : {time%60:02}s)
        </p>
    """, unsafe_allow_html = True)


def FiltrationResult(countNP):
    st.markdown(f"""
        <p class = 'text'>
            Nanoparticles after filtration: <b>{countNP}</b>
        </p>
    """, unsafe_allow_html = True)


def LabelUploderFileCVAT():    
    st.markdown(f"""
        Import <a href='https://app.cvat.ai/'>CVAT</a> data to calculate statistics (format 'CVAT for images 1.1')
    """, unsafe_allow_html = True)


def AboutSectionParticleParams():
    st.markdown(f"""
        <p class = 'text center'>
            The main parameters of nanoparticles can be represented as primary values: 
            the average diameters, its deviations, or a histogram of the diameters distribution. 
            <br>Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
            which can be normalized to the area of the SEM image.
        </p>
    """, unsafe_allow_html = True)

    
def MaterialDensity(value):
    st.markdown(f"""
        <div class = 'text' style = "font-size: 16px;">
            Particles material density: <b>{value:.2e} ng/nm<sup>3</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def EstimatedScale(scale):
    st.markdown(f"""
        <div class = 'text'>
            Estimated scale: <b>{scale:.3f} nm/px</b> 
        </div>
    """, unsafe_allow_html = True)


def NameMaterial(key, options, density):
    if key != 4:
        st.markdown(f"""
            <div class = 'text'>
                Material: <b>{options[key]}</b> 
            </div>
        """, unsafe_allow_html = True)
    else:
        st.markdown(f"""
            <div class = 'text'>
                Material: <b>{options[key]} ({density:.2e} ng/nm<sup>3</sup>)</b> 
            </div>
        """, unsafe_allow_html = True)


def Quantity(allNP, currentNP):
    if allNP == currentNP:
        st.markdown(f"""
            <div class = 'text'>
                Quantity: <b>{allNP}</b>
            </div>
        """, unsafe_allow_html = True)
    else:
        st.markdown(f"""
            <div class = 'text'>
                Quantity: <b>{allNP}</b> (includ {currentNP} selected)
            </div>
        """, unsafe_allow_html = True)


def PrimaryParameters(diameters):
    st.subheader("Primary parameters", anchor = False)

    st.markdown(f"""
        <div class = 'text'>
            Average diameter: <b>{np.mean(diameters):.2f} nm</b> 
        </div>
    """, unsafe_allow_html = True)

    st.markdown(f"""
        <div class = 'text'>
            Standart deviation diameters: <b>{np.std(diameters):.2f} nm</b> 
        </div>
    """, unsafe_allow_html = True)


def SecondaryParameters(paramsNP):    
    st.subheader("Secondary parameters", anchor = False) 

    st.markdown(f"""
        <div class = 'text'>
            Mass: <b>{paramsNP["mass"]:0.2e} ng</b> 
        </div>
    """, unsafe_allow_html = True)   

    st.markdown(f"""
        <div class = 'text'>
            Volume: <b>{paramsNP["volume"]:0.2e} nm<sup>3</sup></b> 
        </div>
    """, unsafe_allow_html = True) 

    st.markdown(f"""
        <div class = 'text'>
            Area: <b>{paramsNP["area"]:0.2e} nm<sup>2</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def NormSecondaryParameters(paramsNP):
    st.subheader("Secondary parameters (norm)",
        help = f"Values relative to the surface area is {paramsNP["imageArea"]:.2e} nm²",
        anchor = False
    )                    
                       
    st.markdown(f"""
        <div class = 'text'>
            Norm area: <b>{paramsNP["normArea"]:0.2f}</b> %
        </div>
    """, unsafe_allow_html = True)

    st.markdown(f"""
        <div class = 'text'>
            Norm mass: <b>{paramsNP["normMass"]:0.2e} ng/nm<sup>2</sup></b> 
        </div>
    """, unsafe_allow_html = True)


def AboutSectionSpatialDistribution():
    st.markdown(f"""
        <p class = 'text center'>
            Visual representation of nanoparticle-based statistics in image.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def AboutSectioQuality():
    st.markdown(f"""
        <p class = 'text center'>
            Quality evaluation of the automatically detected nanoparticles 
            based on the Jacquard measure and the expert's manual marking.
            A detailed description is provided in the work on the [2] link below.
        </p>
    """, unsafe_allow_html = True)


def LegendChartQuality():
    st.markdown("""
        <div style="text-align: center;">
            Types particles in chart:<br>
            <span class="particle-label blue">Detect by algorithm</span>
            <span class="particle-label green">Correctly identified by algorithm (TP)</span>
            <span class="particle-label red">Not identified by algorithm (FN)</span>
            <span class="particle-label orange">Identified but not confirmed by expert (FP)</span>
        </div>
    """, unsafe_allow_html=True)


def Guide1():
    st.subheader("Детектирование и фильтрация наночастиц", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class = 'text'>Все дальнейшие шаги выполняются на вкладке «Automatic detection».</p>
            <ul>
                <li>
                    <p class = 'text'>
                        Шаг 1. Загрузка исходного СЭМ-изображения (кнопка «Browse file»).
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Шаг 2. Детектирование наночастиц (кнопка «Nanoparticles detection» становится активной после
                        загрузки изображения). Процесс детектирования занимает некоторое время, в среднем до одной минуты.                     
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Шаг 3. После успешного детектирования производится фильтрация найденных наночастиц
                        (используются параметры по умолчанию). Отфильтрованные частицы отображаются
                        на изображении в виде окружностей.
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Шаг 4. Можно вручную изменять параметры детектирования и фильтрации наночастиц
                        (снять галочку «Use default settings»). <strong>ВАЖНО:</strong> подтверждение параметров 
                        детектирования осуществляется повторным нажатием кнопки «Nanoparticles detection». Параметры
                        фильтрации применяются автоматически.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)

    media_col.markdown(f"""
        <div class = 'text' style = "text-align: center;">
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide2():
    st.subheader("Взаимодейтсвие с результатами детектирования", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class = 'text'>Указанный функционал доступен на вкладке «Automatic detection» после детектирования наночастиц.</p>
            <ul>
                <li>
                    <p class = 'text'>
                        Результаты детектирования можно скачать в нескольких вариантах:
                        (1) Найденные частицы на прозрачном фоне. (2) Найденные частицы, наложенные на исходное изображение.
                        (3) Файл с указанием координат центра и радиуса каждой частицы.
                        Для этого нужно в выпадающем списке «What results should be saved?» выбрать нужный вариант 
                        и нажать кнопку, расположенную правее.
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Если на изображении присутствует мерная шкала и указан её физический размер, масштаб 
                        определяется автоматически. Визуализировать вычисленный масштаб можно с помощью 
                        переключателя «Display scale».
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Режим сравнения в доработке!
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text center'>
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide3():
    st.subheader("Интеграция с CVAT", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>              
            <ul>
                <li>
                    <p class = 'text'>
                        Результаты детектирования можно скачать в формате, поддерживаемом <a href=https://app.cvat.ai/>CVAT</a>.
                        Для этого на вкладке «Automatic detection» после детектирования наночастиц
                        нужно в выпадающем списке «What results should be saved?» выбрать
                        пункт «CVAT task» и нажать кнопку, расположенную правее. Скачанный backup-архив можно 
                        использовать для создания новой задачи CVAT.
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Разметку, полученную в CVAT, можно импортировать на сайт. Для этого сначала необходимо выгрузить
                        из CVAT backup-архив задачи с нужной разметкой. Затем на вкладке «Statistics dashboard» 
                        в выпадающем списке «Which nanoparticles to use» нужно выбрать пункт «Import from CVAT» и 
                        загрузить backup-архив в соответствующее поле. Если все условия выполнены, ниже автоматически 
                        отобразятся все разделы со статистикой.
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Более подробная информация об интеграции с CVAT приведена в 
                        <a href = "https://disk.yandex.ru/i/2U5wgJ8IjskREQ"
                            >расширенном руководстве</a>.
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text center'>
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def Guide4():
    st.subheader("Оценка качества детектирования", anchor = False)
    text_col, media_col = st.columns([1, 1], vertical_alignment = 'center')

    text_col.markdown(f"""
        <div>
            <p class = 'text'>Все дальнейшие шаги выполняются на владке «Statistics dashboard».</p>
            <ul>                    
                <li>
                    <p class = 'text'>
                        В разделе «Quality evaluation» можно получить численную оценку качества детектирования наночастиц.
                        Для этого, в первую очередь, необходим результат автоматического детектирования. Он должен быть
                        либо на вкладке «Automatic detection», либо в виде backup-архива CVAT, который нужно загрузить 
                        в разделе «Global dashboard settings». Далее требуется загрузить файл с экспертной разметкой, 
                        также в формате backup-архива CVAT, в соответствующее поле раздела «Quality evaluation». Если 
                        все условия выполнены, ниже отобразится качество в процентах. Подробно процедура оценки качества 
                        описана в работе [2].
                    </p>
                </li>
                <li>
                    <p class = 'text'>
                        Дополнительно можно визуализировать результат оценки качества детектирования. Для этого
                        переключите тумблер «Display nanoparticles». В результате ниже появится интерактивный график,
                        на котором будут отмечены наночастицы четырёх типов: "Синие" - это автоматически детектированные
                        частицы, которые сопоставлены с "зелёными" наночастицами, отмеченными экспертом (TP).
                        "Красные" — это наночастицы, которые были помечены экспертом, но не были детектированы 
                        автоматически (FN). "Жёлтые" - это автоматически детектированные наночастицы, которые не были 
                        подтверждены экспертом (FP).
                    </p>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)
       
    media_col.markdown(f"""
        <div class = 'text' style = "text-align: center;">
            A video guide will be added here soon!
        </div>
    """, unsafe_allow_html = True)


def HowCite():
    tempCol = st.columns([0.8, 0.2], vertical_alignment = 'center')

    tempCol[0].markdown("""
        <div class = 'cite'> <b>How to cite</b>:
            <ul>
                <li> <p class = 'cite'>
                    [1] An article about this site will be published soon, don't miss it!
                </p> </li>
                <li> <p class = 'cite'>
                    [2] Automated Recognition of Nanoparticles in Electron Microscopy Images of Nanoscale Palladium Catalysts.
                    Boiko D.A., Sulimova V.V., Kurbakov M.Yu. [et al.] 
                    // Nanomaterials. 2022. Vol. 12, No. 21. Pp. 3914. 
                    DOI: <a href=https://www.mdpi.com/2079-4991/12/21/3914>10.3390/nano12213914</a>.
                </p> </li>
                <li> <p class = 'cite'>
                    [3] Determining the Orderliness of Carbon Materials with Nanoparticle Imaging and Explainable Machine Learning. 
                    Kurbakov M.Yu., Sulimova V.V., Kopylov A.V. [et al.]
                    // Nanoscale. 2024. Vol. 16, No. 28. Pp. 13663-13676. 
                    DOI: <a href=https://pubs.rsc.org/en/content/articlelanding/2024/nr/d4nr00952e>10.1039/d4nr00952e</a>.
                </p> </li>                
                <li> <p class = 'cite'>
                    [4] Interpretable Graph Methods for Determining Nanoparticles Ordering in Electron Microscopy Images.
                    Kurbakov M.Yu., Sulimova V.V., Seredin O.S., Kopylov A.V. // Computer Optics. 2025. Vol. 49, No 3. Pp. 470-479.
                    DOI: <a href=https://computeroptics.ru/eng/KO/Annot/KO49-3/490313e.html>10.18287/2412-6179-CO-1568</a>.
                </p> </li>
            </ul>
        </div>
    """, unsafe_allow_html = True)

    tempCol[1].image(r"./nano-website/content/qr-code.svg",
        caption = "Web Nanoparticles QR-code",
        use_container_width = True
    )   


def Footer():
    st.markdown(f"""
        <div class = 'footer'>
            Laboratory of Cognitive Technologies and Simulating Systems,
            Tula State University © {datetime.datetime.now().year} (E-mail: nanoweb.assist@gmail.com)
        </div>
    """, unsafe_allow_html = True)