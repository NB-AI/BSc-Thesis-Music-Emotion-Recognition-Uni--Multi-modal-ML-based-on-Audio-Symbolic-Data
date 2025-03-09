def check(user_sys, list_of_modules, multimodal_prediction=False):

    if multimodal_prediction:
        version_dict = {
            'sys': '3.7.3',
            'os':'file directly in folder python3.7',    
            'numpy':'1.21.6',    
            'pandas':'2.0.3',
            'glob':'file directly in folder python3.7',
            'librosa':'0.10.0.post2',
            'scipy':'1.10.1',   
            'torch':'2.0.1+cu117',
            'torcheval':'0.0.7',
            'collections':'file directly in folder python3.7',
            'sklearn':'1.3.0'}
 
    else:
        version_dict = {
            'os':'file directly in folder python3.7',
            'random':'file directly in folder python3.7',
            'subprocess':'file directly in folder python3.7',
            'webbrowser':'file directly in folder python3.7',
            'glob':'file directly in folder python3.7',
            'collections':'file directly in folder python3.7',
            'math':'file directly in folder python3.7',
            'pickle':'file directly in folder python3.7',
            'ast':'file directly in folder python3.7',
            'contextlib':'file directly in folder python3.7',
            'inspect':'file directly in folder python3.7',
            'traceback':'file directly in folder python3.7',
            'logging':'file directly in folder python3.7',
            'pathlib':'file directly in folder python3.7',
            'urllib':'file directly in folder python3.7',
            'time':'3.1.2',
            'magic': '0.4.27',
            'sys': '3.7.3',
            'pandas':'1.3.5',
            'numpy':'1.21.6',
            're':'2.2.1',
            'requests':'2.28.1',
            'bs4':'4.9.1',
            'selenium':'4.1.3',
            'virustotal_python':'0.2.0',
            'pprint':'0.40.0',
            'shutil':'1.0.0',
            'mido':'1.2.10',
            'pydub':'0.25.1',
            'essentia':'2.1-beta6-dev',
            'json':'2.0.9',
            'librosa':'0.8.1',
            'scipy':'1.7.3',
            'opensmile':'2.4.1',
            'miditoolkit':'0.1.16',
            'music21':'7.3.3',
            'torch':'1.8.1+cu102',
            'sklearn':'1.0.2',
            'fairseq':'0.10.2'}

    user_sys_version = user_sys.version 

    sys_str = 'sys'

    list_of_modules.append(user_sys)

    for module in list_of_modules:

        str_module = str(module).split(' ')[1][1:-1]

        if str_module in version_dict.keys():

            original_version = version_dict[str_module]


            try:
                user_version = module.__version__

            except:
                try:
                    user_version = module.version

                except:
                    user_version = ' is not clearly visible. Go to your used python folder path .../python3.7/site-packages to investigate the version of your package. Also possible: have a look at the shell commands "pip show module_name" and "apt show module_name"'

    
            if user_version == original_version or (str_module==sys_str and original_version in user_version):
                print(f'Your version identical with original version \n-> {str_module} \n-> {original_version}')

            elif original_version=='file directly in folder python3.7':
                print(f'For \n{str_module} \npython version needs to fit \n-> original version {version_dict[sys_str]} \n-> your version {user_sys_version}')            

            else:
                print(f'Possibly different versions: \n-> {str_module} \n-> original version {original_version} \n-> your version {user_version}')

            
        else:
            print(f'No match for your module \n{str_module} \nfound in the requirements.')
                        
        print('\n')

    return
