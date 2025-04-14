# Run application
# streamlit run .\nano-website\app.py --server.enableXsrfProtection false

try:
    import streamlit as st

    import io, csv
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageDraw
    import time, datetime

    import style, autoscale, nanoStatistics
    import ExponentialApproximation as EA

    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    import traceback
    

    ### Function ###
    
    help_str = "be added soon"

    def load_default_session_state(_dispToast = False):
        if _dispToast:
            st.toast('Default configuration loaded!')

        st.session_state['rerun'] = False

        st.session_state['imageUpload'] = False
        st.session_state['uploadedImage'] = None

        st.session_state['settingDefault'] = True
        st.session_state['param1'] = 10
        st.session_state['param2'] = (1.0, 5.0)
        st.session_state['param3'] = 0.65
        st.session_state['param-pre-1'] = 10
        st.session_state['parallel'] = True
    
        st.session_state['detected'] = False
        st.session_state['BLOBs'] = None
        st.session_state['BLOBs_params'] = None
        st.session_state['BLOBs_filter'] = None
        st.session_state['sizeImage'] = None
        st.session_state['detectedParticles'] = 0    
        st.session_state['filteredParticles'] = 0 
        st.session_state['imageBLOBs'] = None
        st.session_state['timeDetection'] = None

        st.session_state['comparison'] = False
    
        st.session_state['scale'] = None
        st.session_state['scaleData'] = None
        st.session_state['displayScale'] = False

        st.session_state['chartRange'] = st.session_state['param2']
        st.session_state['distView'] = False
        st.session_state['normalize'] = False
        st.session_state['recalculation'] = False

        st.session_state['calcStatictic'] = False


    def get_session_state():
       return str(st.session_state)

    @st.dialog("Something went wrong...")
    def dialog_exception(_exception):
        st.write("""
            An error occurred while the application was running.
            *The latest detection and marking results are saved.*
            Refresh the site with the "Rerun page" button below
            (if you refresh the page through the browser, some data may not be saved).
        """)

        err_msg = "!!! CRASH\t" +\
            f"{datetime.datetime.now().ctime()} \n\t" + \
            f"{get_session_state()} \n\t" + \
            f"{str(_exception)} \n\t" + \
            f"{traceback.format_exc()} \n" + \
            "CRASH !!!\n"
        print(err_msg)

        if st.button("Rerun page"):
            st.session_state['rerun'] = True
            st.rerun()

        with st.expander("Info for developers", expanded = False, icon = ":material/app_registration:"):
            st.write(traceback.format_exc())

    # not used
    def send_email_report(_email, _password, _text, _uploadedFile):        
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        # Создание MIME-сообщения
        msg = MIMEMultipart()
        msg["From"] = _email
        msg["To"] = _email
        msg["Subject"] = f"Crash on {datetime.datetime.now().ctime()}"

        # Добавление текста письма
        msg.attach(MIMEText(_text, "plain"))

        # Добавление вложения
        part = MIMEBase("application", "octet-stream")
        part.set_payload(_uploadedFile.getvalue())
        encoders.encode_base64(part)  # Кодировка в Base64
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={_uploadedFile.name}",  # Имя файла
        )
        msg.attach(part)

        # Отправка через SMTP (для Gmail)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(_email, _password)
            server.sendmail(_email, _email, msg.as_string())



    ### Main app ###
    
    # Loading CSS styles
    style.set_style()
    
    # Initial loading of session states
    if 'rerun' not in st.session_state:
        load_default_session_state()
    elif st.session_state['rerun']:
        load_default_session_state(True)
    
    ## Header
    st.markdown("<div class = 'header'>WEB NANOPARTICLES</div>", unsafe_allow_html = True)
    
    st.markdown("""
        <div class = 'about'>
            Hello! It is an interactive tool for processing images from a scanning electron microscope (SEM).
            <br>It will help you to detect palladium nanoparticles in the image and calculate their statictics.
        </div>
    """, unsafe_allow_html = True)

    st.markdown("""
        <div style = "padding-bottom: 25px" class = 'about'>
            Examples of SEM images for analysis are <a href=https://doi.org/10.6084/m9.figshare.11783661.v1>here</a>.
        </div>
    """, unsafe_allow_html = True)


    ## Main content area
    tabDetect, tabLable, tabInfo = st.tabs([
        "Automatic detection nanoparticles",
        "Manual labeling nanoparticles",
        "Statistics dashboard"
    ])

    ## TAB 1
    with tabDetect:
        colImage, colSetting = st.columns([6, 2])

        imagePlaceholder = None

        # Viewing images
        with colImage:
            st.subheader("Upload SEM image")
            uploadedImage = st.file_uploader("Choose an SEM image", type = ["png", "jpg", "jpeg", "tif"])

            if uploadedImage is None:
                load_default_session_state()
            else:
                st.session_state['imageUpload'] = True
                crsImage = Image.open(uploadedImage)
                grayImage = np.array(crsImage.convert('L'), dtype = 'uint8')               

                if (not np.array_equal(st.session_state['uploadedImage'], grayImage)):
                    st.session_state['uploadedImage'] = grayImage
                    st.session_state['detected'] = False
                    st.session_state['comparison'] = False
                    st.session_state['calcStatictic'] = False
                    st.session_state['imageBLOBs'] = None
        
        
                imagePlaceholder = st.empty()
                if (st.session_state['imageBLOBs'] is not None):
                    imagePlaceholder.image(st.session_state['imageBLOBs'], use_container_width = True)

        # END left side        

        # Detection settings and results
        with colSetting:        
            st.checkbox("Use default settings?",
                disabled = not st.session_state['imageUpload'],
                key = 'settingDefault',
                help = "You need to upload an SEM image"
            )
    
            # Preprocessing and detection settings       
            st.subheader("Detection settings",
                help = "After changing the value, it is necessary to re-detect the nanoparticles"
            )    
        
            with st.container(border = True):
                st.slider("Nanoparticle brightness",
                    key = 'param-pre-1',
                    disabled = st.session_state['settingDefault'],
                    help = "The average brightness of nanoparticles and its surroundings in the image"
                )

                st.checkbox("Parallel computing", key = 'parallel', disabled = st.session_state['settingDefault'],)
        
            pushDetectButton = st.button("Nanoparticles detection",
                use_container_width = True,
                disabled = not st.session_state['imageUpload'],
                help = "You need to upload an SEM image"
            )
                
            # Detecting
            if pushDetectButton:
                with st.spinner("Nanoparticles detection", show_time = True):
                    timeStart = time.time()
                
                    st.session_state['scale'], st.session_state['scaleData'] = autoscale.estimateScale(grayImage)
               
                    currentImage = np.copy(grayImage) 
         
                    lowerBound = autoscale.findBorder(grayImage)        
                    if (lowerBound is not None):
                        currentImage = currentImage[:lowerBound, :]

                    st.session_state['sizeImage'] = currentImage.shape

                    params = {
                        "sz_med" : 4,   # для предварительной обработки
                        "sz_th":  4,    # для предварительной обработки (не надо равное 5 - кружки получаются большие) 
                        "thr_br": float(st.session_state['param-pre-1']),   # порог яркости для отбрасывания лок. максимумов (Prefiltering)
                        "min_dist": 4,  # минимальное расстояние между локальными максимумами при поиске локальных максимумов (Prefiltering)
                        "wsize": 9,     # размер окна аппроксимации
                        "rs": np.arange(1.0, 7.0, 0.1), # возможные радиусы наночастиц в пикселях
                        "best_mode": 3, # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
                        "msk": 3,       # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
                        "met": 'exp',   # аппроксимирующая функция "exp" или "pol" 
                        "npar": 2       # число параметров аппроксимации
                    }

                    # вычисляется только один раз при первом запуске детектирования
                    helpMatrs, xy2 = EA.CACHE_HelpMatricesNew(params["wsize"], params["rs"])

                    # вычисляется только один раз для одного и того же изображения
                    lm, currentImage = EA.CACHE_PrefilteringPoints(currentImage, params)

                    # вычисляется только один раз для одного набора параметров
                    if st.session_state['parallel']:
                        BLOBs, BLOBs_params = EA.CACHE_ExponentialApproximationMask_v3_parallel(
                            currentImage,
                            lm,
                            xy2,
                            helpMatrs,
                            params
                        )
                    else:
                        BLOBs, BLOBs_params = EA.CACHE_ExponentialApproximationMask_v3(
                            currentImage,
                            lm,
                            xy2,
                            helpMatrs,
                            params
                        )
            
                    st.session_state['BLOBs'] = BLOBs
                    st.session_state['BLOBs_params'] = BLOBs_params
                    st.session_state['detected'] = True
                
                    st.session_state['timeDetection'] = int(np.ceil(time.time() - timeStart))
                    st.session_state['detectedParticles'] = BLOBs.shape[0]
        
            # Detection results
            if st.session_state['detected']:
                time = st.session_state['timeDetection']
                st.markdown(f"""
                    <p class = 'text'>
                        Nanoparticles detected: <b>{st.session_state['detectedParticles']}</b> ({time//60}m : {time%60:02}s)
                    </p>""", unsafe_allow_html = True
                )

            # Warning about not correctly detection results 
            if (st.session_state['detected'] and st.session_state['detectedParticles'] < 1):            
                st.warning("""
                    Nanoparticles not found!
                    Please change the detection settings or upload another SEM image!
                """, icon = ":material/warning:")
        
            # Action with correctly detection results
            if (st.session_state['detected'] and st.session_state['detectedParticles'] > 0):
                # Filtration settings
                st.subheader("Filtration settings",
                    help = "Choosing among the detected nanoparticles those that meet the relevant criteria"
                )
            
                with st.container(border = True):
                    st.slider("Nanoparticle center brightness",
                        key = 'param1',
                        value = 10,
                        disabled = st.session_state['settingDefault'],
                        help = "Brightness in the central pixel of the nanoparticle"
                    )

                    st.slider("Range of nanoparticle radii, nm",
                        key = 'param2',
                        value = (1.0, 5.0),
                        min_value = 1.0,
                        step = 0.1,
                        max_value = 7.0,
                        disabled = st.session_state['settingDefault']
                    )

                    st.slider("Nanoparticle reliability",
                        key = 'param3',
                        value = 0.65,
                        min_value = 0.0,
                        step = 0.01,
                        max_value = 1.0,
                        disabled = st.session_state['settingDefault'],
                        help = "The higher the reliability, the clearer the nanoparticle is against the background of the image"
                    )
                
                params_filter = {
                    "thr_c0": st.session_state['param1'],
                    "min_thr_r": st.session_state['param2'][0],   
                    "max_thr_r": st.session_state['param2'][1], 
                    "thr_error": 1 - st.session_state['param3'], 
                }
            
                # Filtering
                st.session_state['BLOBs_filter'], _ = EA.my_FilterBlobs_change(
                    st.session_state['BLOBs'],
                    st.session_state['BLOBs_params'],
                    params_filter
                )
                st.session_state['filteredParticles'] = st.session_state['BLOBs_filter'].shape[0]

                # Info about filtered nanoparticles
                st.markdown(f"""
                    <p class = 'text'>
                        Nanoparticles after filtration: <b>{st.session_state['filteredParticles']}</b>
                    </p>""", unsafe_allow_html = True
                )

                # Slider for comparing the results before and after detection
                st.toggle("Comparison mode", key = 'comparison', disabled = True, help = help_str)

                # Displaying the scale
                st.toggle("Display scale", key = 'displayScale', help = help_str)

                # Saving
                safeImgCol, safeBLOBCol = st.columns(2)
                        
                # Saving image
                with safeImgCol:
                    temp = Image.new(mode = "RGBA", size = st.session_state['sizeImage'][::-1])
                    draw = ImageDraw.Draw(temp)
                    for BLOB in st.session_state['BLOBs_filter']:                
                        y, x, r = BLOB          
                        draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0))

                    file = io.BytesIO()
                    temp.save(file, format = "PNG")

                    st.download_button(
                        label = "Download nanoparticles image",
                        data = file.getvalue(),
                        file_name = Path(uploadedImage.name).stem + "-image.tif",
                        use_container_width  = True,
                        help = help_str
                    )
            
                # Saving coords
                with safeBLOBCol:
                    file = io.StringIO()

                    if st.session_state['scale'] is not None: 
                        csv.writer(file).writerow([f"{st.session_state['scale']:.3} nm/px"])
                    
                    csv.writer(file, delimiter = ';').writerow(['coord y, px', 'coord x, px', 'radius, px'])
                    csv.writer(file, delimiter = ';').writerows(st.session_state['BLOBs_filter'])

                    st.download_button(
                        label = "Download nanoparticles *.csv",
                        data = file.getvalue(),
                        file_name = Path(uploadedImage.name).stem + "-particles.csv",
                        use_container_width  = True,
                        help = help_str
                    )
        # END right side

        # Display source image by st.image
        if (st.session_state['imageUpload']):
            viewImage = crsImage

            if (st.session_state['filteredParticles'] > 1):
                if (st.session_state['detected'] and not st.session_state['comparison']):
                        imageBLOBs = crsImage.convert("RGBA")
                        draw = ImageDraw.Draw(imageBLOBs)
                        for BLOB in st.session_state['BLOBs_filter']:                
                            y, x, r = BLOB          
                            draw.ellipse((x-r, y-r, x+r, y+r), outline = (0, 225, 0, 200))

                        viewImage = imageBLOBs
        
            st.session_state['imageBLOBs'] = viewImage
            imagePlaceholder.image(
                viewImage,
                use_container_width = True
            )


    ## TAB 2
    with tabLable:
        st.warning("""
                The section is temporarily unavailable, but it will appear soon!
            """, icon = ":material/warning:")


    ## TAB 3
    with tabInfo:    
        heightCol = 550
        marginChart = dict(l=10, r=10, t=40, b=5)

        if (not st.session_state['detected']):
            st.warning("""
                Nanoparticle detection is necessary to calculate their statistics.
                Please go to "Detection" tab.
            """, icon = ":material/warning:")
        elif (st.session_state['filteredParticles'] < 10):                  
            st.warning("""
                Nanoparticles after detection and filtration are less than 10! 
                Please go to the "Detection" tab and change the detection, filtering settings or upload another SEM image!
            """, icon = ":material/warning:")
        else:
            with st.expander("Global dashboard settings", expanded = not st.session_state['calcStatictic'], icon = ":material/rule_settings:"):

                option_map = {
                    0: "Automatically detected (AD)",
                    1: "Manual labeled (ML)",
                    2: "Union of AD and ML",
                    3: "Intersection of AD and ML",
                }

                selection = st.radio(
                    "Which nanoparticles to use?",
                    index = 0,
                    options = option_map.keys(),
                    format_func = lambda option: option_map[option],
                    horizontal = True,
                    disabled = True
                )

                def temp_fun():                
                    st.session_state['calcStatictic'] = True
          
                temp_push = st.button("Calculate statistics", key = 'right_button', on_click = temp_fun)           

        if (st.session_state['calcStatictic']):
            if st.session_state['scale'] is None:
                radius_nm = st.session_state['BLOBs_filter'][:, 2]            
                fullDist, minDist = nanoStatistics.euclideanDistance(st.session_state['BLOBs_filter']) 
            else:            
                radius_nm = st.session_state['BLOBs_filter'][:, 2] * st.session_state['scale']
                fullDist, minDist = nanoStatistics.euclideanDistance(st.session_state['BLOBs_filter'] * st.session_state['scale']) 

            boolIndexFiltringBLOBs = None       
            
            additionalSTR = ""

            with st.expander("Particle parameters", expanded = True, icon = ":material/app_registration:"):
                st.markdown(f"""
                    <p class = 'text'>
                        The main parameters of nanoparticles can be represented as primary values: 
                        the average radius, its deviations, or a histogram of the radius distribution. 
                        Or secondary values: particle mass, volume, area (projection onto a two-dimensional plane), 
                        which can be normalized to the area of the SEM image.
                    </p>""", unsafe_allow_html = True
                )

                db11, db12, db13 = st.columns([4, 4, 4])            

                minRadius, maxRadius = st.session_state['chartRange']
                boolIndexFiltringBLOBs = (radius_nm >= minRadius) & (radius_nm <= maxRadius)
                radiusFiltered = radius_nm[boolIndexFiltringBLOBs]
                additionalSTR = f" (from {minRadius:.1f} to {maxRadius:.1f})"

                # Particle size distribution
                with db11.container(border = True, height = heightCol):
                    left, rigth = st.columns([7, 1], vertical_alignment = 'center')
                    left.subheader("Distribution of particle radius" + additionalSTR)

                    buttonPlaceholder = None

                    with rigth.popover("", icon=":material/settings:"):
                        minVal, maxVal = st.session_state['param2']
                        st.slider("Range of nanoparticle radii, nm",
                            key = 'chartRange',
                            value = (minVal, maxVal),
                            min_value = minVal,
                            step = 0.1,
                            max_value = maxVal
                        )

                        st.checkbox("Display distribution function?",
                            key = 'distView',
                            help = help_str
                        )

                        st.checkbox("Normalize the vertical axis?",
                            key = 'normalize',
                            help = help_str
                        )

                        st.checkbox("Recalculation parameters for range?",
                            key = 'recalculation',
                            help = help_str
                        )
                        
                        buttonPlaceholder = st.empty()
                                        
                    start = 1; step = 0.25; end = maxRadius + step
                    counts, bins = np.histogram(radiusFiltered, bins = np.arange(start, end, step, dtype = float))
                                        
                    name_x = "Radius, nm"
                    temp = [[float(i), float(i+step)] for i in bins]
                    fraction = counts / np.sum(counts) * 100
                    if st.session_state['normalize']:
                        bar_y = fraction
                        name_y = "Particles fraction, %"
                        hover_y = "%{y:.1f}% (%{customdata[2]:d})"
                        dataChart = [list(pair) for pair in zip(temp, fraction)]
                        customDataChart = list(zip(bins, bins + step, counts))
                    else:
                        bar_y = counts
                        name_y = "Particles counts"
                        hover_y = "%{y:d} (%{customdata[2]:.1f}%)"
                        dataChart = [list(pair) for pair in zip(temp, counts)]
                        customDataChart = list(zip(bins, bins + step, fraction))
                                            
                    file = io.StringIO()
                    csv.writer(file, delimiter = ';').writerow([name_x, name_y])
                    csv.writer(file, delimiter = ';').writerows(dataChart)

                    buttonPlaceholder.download_button(
                        label = "Download data chart *.csv",
                        data = file.getvalue(),
                        file_name = Path(uploadedImage.name).stem + "-dist-radius.csv",
                        use_container_width  = True,
                        help = help_str
                    )

                    fig = go.Figure()

                    fig = fig.add_trace(go.Bar(
                        x = 0.5 * (bins[:-1] + bins[1:]),
                        y = bar_y,
                        customdata = customDataChart,
                        hovertemplate = (
                            "Radius: [%{customdata[0]}, %{customdata[1]}) nm<br>"
                            "Particls: " + hover_y +
                            "<extra></extra>"
                        )
                    ))

                    if st.session_state['distView']:
                        mu = np.mean(radiusFiltered)
                        sigma = np.std(radiusFiltered)

                        dist_x = np.arange(start, end, step * 0.1, dtype = float)

                        fig.add_trace(go.Scatter(
                            x = dist_x, 
                            y = np.max(bar_y) / (sigma * np.sqrt(2*np.pi)) * np.exp(-(dist_x - mu)**2 / (2 * sigma**2)),
                            mode = 'lines',
                            name = '',
                            hoverinfo = 'skip'
                        ))         

                    fig.update_layout(
                        margin = marginChart,
                        xaxis_title_text = name_x,
                        yaxis_title_text = name_y,
                        showlegend = False
                    )

                    fig.update_xaxes(
                        tickmode = 'linear',
                        dtick = step,
                        tick0 = start,
                        tickwidth = 2,
                        showgrid = True,
                        gridwidth = 1,
                    )
                    
                    st.plotly_chart(fig, use_container_width = True)
                # END db11

                # Nanoparticle parameters
                with db12.container(border = True, height = heightCol):                        
                    # for Pd (palladium)
                    materialDensity = 12.02 * 10**-15 # nanogram / nanometer   

                    if st.session_state['scale'] is None:
                        st.markdown(f"""
                            <div class = 'text'>
                                The image scale could not be determined automatically!
                                Using default scale: 1 nm/px
                            </div>""", unsafe_allow_html = True)
                    else:                    
                        st.markdown(f"""
                            <div class = 'text'>
                                Estimated scale: <b>{st.session_state['scale']:.3} nm/px</b> 
                            </div>""", unsafe_allow_html = True)

                        if st.session_state['recalculation']:
                            radius_nm = radiusFiltered
                        
                        st.markdown(f"""
                            <div class = 'text'>
                                Particles number: <b>{st.session_state['filteredParticles']}</b> 
                            </div>""", unsafe_allow_html = True)

                        st.subheader("Primary parameters")                    
                        st.markdown(f"""
                            <div class = 'text'>
                                Average radius: <b>{np.mean(radius_nm):0.3} nm</b> 
                            </div>""", unsafe_allow_html = True)
                        st.markdown(f"""
                            <div class = 'text'>
                                Standart deviation radius: <b>{np.std(radius_nm):0.3} nm</b> 
                            </div>""", unsafe_allow_html = True)


                        st.subheader("Secondary parameters")  
                        volumeParticls = 4 / 3 * np.pi * radius_nm ** 3
                        areaParticls =  np.sum(2 * np.pi * radius_nm ** 2)
                        massParticls = np.sum(volumeParticls * materialDensity)

                        st.markdown(f"""
                            <div class = 'text'>
                                Mass: <b>{massParticls:0.2e} nanograms</b> 
                            </div>""", unsafe_allow_html = True)                        
                        st.markdown(f"""
                            <div class = 'text'>
                                Volume: <b>{np.sum(volumeParticls):0.2e} nanometers<sup>3</sup></b> 
                            </div>""", unsafe_allow_html = True)                        
                        st.markdown(f"""
                            <div class = 'text'>
                                Area: <b>{areaParticls:0.2e} nanometers<sup>2</sup></b> 
                            </div>""", unsafe_allow_html = True)
                    

                        st.subheader("Secondary parameters (norm)", help = help_str)
                        imageArea = st.session_state['sizeImage'][0] * st.session_state['sizeImage'][1] * st.session_state['scale'] ** 2
                    
                        st.markdown(f"""
                            <div class = 'text'>
                                Mass: <b>{massParticls/imageArea:0.2e} nanograms/nanometers<sup>2</sup></b> 
                            </div>""", unsafe_allow_html = True)                        
                        st.markdown(f"""
                            <div class = 'text'>
                                Volume: <b>{np.sum(volumeParticls)/imageArea:0.2e} nanometers</b> 
                            </div>""", unsafe_allow_html = True)                        
                        st.markdown(f"""
                            <div class = 'text'>
                                Area: <b>{areaParticls/imageArea*100:0.2f}</b> %
                            </div>""", unsafe_allow_html = True)
                # END db12

                # Heatmap of particle count
                with db13.container(border = True, height = heightCol):
                    stepSize = 10
                    uniformityMap = nanoStatistics.uniformity(
                        st.session_state['BLOBs_filter'][boolIndexFiltringBLOBs] if st.session_state['recalculation'] else st.session_state['BLOBs_filter'],
                        st.session_state['sizeImage'], stepSize
                    )

                    fig = px.imshow(uniformityMap)

                    titleChart = "Heatmap of particle count";
                    if st.session_state['recalculation']:
                        titleChart += additionalSTR

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = titleChart, font = dict(size=27)),
                        xaxis_title_text = f'Width image, {stepSize}*px',
                        yaxis_title_text = f'Height image, {stepSize}*px',
                        showlegend = False
                    )

                    st.plotly_chart(fig, use_container_width = True)
                # END db13

            with st.expander("Some information charts ", icon = ":material/data_thresholding:"):
                st.markdown(f"""
                    <p class = 'text'>
                        Visual representation of nanoparticle-based statistics in an image
                    </p>""", unsafe_allow_html = True)


                db21, db22, db23 = st.columns([1, 1, 1])

                # ?
                with db23.container(border = True, height = heightCol):
                    st.markdown(f"""
                        <div class = 'about' style = "text-align: center;">
                            Any other information or chart
                        </div>""", unsafe_allow_html = True)
                # END db23

                # ?
                with db21.container(border = True, height = heightCol):                
                    tempDist = minDist[boolIndexFiltringBLOBs] if st.session_state['recalculation'] else minDist

                    fig = ff.create_distplot(
                        [tempDist], [''], bin_size = 2, histnorm = 'probability',
                        colors = ['green'], show_curve = False, show_rug = False
                    )

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Distance to the nearest nanoparticle" + additionalSTR, font = dict(size=27)),
                        xaxis_title_text = 'Distance to the nearest nanoparticle, nm',
                        yaxis_title_text = 'Particle fraction',
                        showlegend = False
                    )
                    
                    fig.update_xaxes(range = [np.floor(tempDist.min()), np.ceil(tempDist.max())],
                        showgrid = True, gridwidth = 0.5, gridcolor = '#606060'
                    )
                    fig.update_traces(hoverinfo = "x", hovertemplate = "distanse: %{x:.2} nm")

                    st.plotly_chart(fig, use_container_width = True)
                # END db21

                # ?
                with db22.container(border = True, height = heightCol):                
                    x = np.arange(5,100,1)
                    temp_db23 = nanoStatistics.thresholdDistance(x, fullDist[boolIndexFiltringBLOBs])

                    fig = px.bar(x = x, y = temp_db23)

                    fig.update_layout(
                        margin = marginChart,
                        title = dict(text = "Distance to the nearest nanoparticle" + additionalSTR, font = dict(size=27)),
                        xaxis_title_text = 'Nanoparticle neighborhood size, nm',
                        yaxis_title_text = 'Average density of nanoparticles in neighborhood',
                        showlegend = False
                    )
                    
                    fig.update_xaxes(range = [np.floor(x.min()), np.ceil(x.max())],
                        showgrid = True, gridcolor = '#606060'
                    )
                    fig.update_traces(width = 0.5, hovertemplate = "?: %{x:.2}")

                    st.plotly_chart(fig, use_container_width = True)
                # END db22

    st.markdown("""
        <div class = 'cite'> <b>How to cite</b>:
            <ul>
                <li> <p class = 'cite'>
                    Automated Recognition of Nanoparticles in Electron Microscopy Images of Nanoscale Palladium Catalysts.
                    <br> D. A. Boiko, V. V. Sulimova, M. Yu. Kurbakov [et al.] 
                    // Nanomaterials. – 2022. – Vol. 12, No. 21. – P. 3914. 
                    – DOI <a href=https://www.mdpi.com/2079-4991/12/21/3914>10.3390/nano12213914</a>
                </p> </li>
                <li> <p class = 'cite'>
                    Determining the Orderliness of Carbon Materials with Nanoparticle Imaging and Explainable Machine Learning. 
                    <br> M. Yu. Kurbakov, V. V. Sulimova, A. V. Kopylov [et al.] 
                    // Nanoscale. – 2024. – Vol. 16, No. 28. – P. 13663-13676. 
                    – DOI <a href=https://pubs.rsc.org/en/content/articlelanding/2024/nr/d4nr00952e>10.1039/d4nr00952e</a>.
                </p> </li>
                <li> <p class = 'cite'>
                    An article about calculating the mass of nanoparticles will be published soon, don't miss it!
                </p> </li>
            </ul>
        </div>""", unsafe_allow_html = True)

    with st.expander("User's Guide", expanded = False, icon = ":material/not_listed_location:"):
        st.subheader("Nanoparticles detection and filtration")
        st.markdown("""
        <div>
            <ul>
                <li>
                    <p class = 'cite'>
                        123
                    </p>
                </li>
                <li>
                    <p class = 'cite'>
                        456
                    </p>
                </li>
                <li>
                    <p class = 'cite'>
                        789
                    </p>
                </li>
            </ul>
        </div>""", unsafe_allow_html = True)

        st.subheader("Calculating statistics")
        st.markdown("""
        <div>
            <ul>
                <li>
                    <p class = 'cite'>
                        aaa
                    </p>
                </li>
                <li>
                    <p class = 'cite'>
                        bbb
                    </p>
                </li>
                <li>
                    <p class = 'cite'>
                        ccc
                    </p>
                </li>
            </ul>
        </div>""", unsafe_allow_html = True)

    st.markdown("""
        <div class = 'footer'>
            Laboratory of Cognitive Technologies and Simulating Systems, Tula State University © 2025 (E-mail: muwsik@mail.ru)
        </div>""", unsafe_allow_html = True)

except Exception as exc:
    dialog_exception(exc)