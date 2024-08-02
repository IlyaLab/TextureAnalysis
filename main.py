from subsets import *
from FeatureExtraction import *
from sort_tiles import *
from ML import *


def check_folder_contents(src):
    # checks to see if the directories already have content. If so, then clears it to redo sorting functions without doubling up on tiles or sample data.
    try:
        for filename in os.listdir(src):
            filepath = os.path.join(src, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)
    except:
        pass


def init_subdirs(MSI_dir, MSS_dir, sample_size=5000, n_jobs=1, validation=True, subsets=True):
    # creates the save path for the results of the texture analysis - found in same directory as code
    if os.path.exists(os.path.join(os.getcwd(),'texture_analysis_results')) == False:
        os.mkdir('texture_analysis_results')
    save_path = os.path.join(os.getcwd(),'texture_analysis_results')

    # creates directories for MSI and MSS texture results
    if os.path.exists(os.path.join(save_path, 'MSI')) == False:
        os.mkdir(os.path.join(save_path, 'MSI'))
    else:
        check_folder_contents(os.path.join(save_path, 'MSI'))
    if os.path.exists(os.path.join(save_path, 'MSS')) == False:
        os.mkdir(os.path.join(save_path, 'MSS'))
    else:
        check_folder_contents(os.path.join(save_path, 'MSS'))

    # creates the directories to be used by the subsets in multiprocessing - found in same directory as tiles
    MSI_subsets_dir = create_subsets(MSI_dir, 'MSI_subsets', sample_size=sample_size, n_jobs=n_jobs)
    MSS_subsets_dir = create_subsets(MSS_dir, 'MSS_subsets', sample_size=sample_size, n_jobs=n_jobs)

    # sorts tiles into their TCGA labelled samples
    sort_tiles(MSI_subsets_dir,  MSS_subsets_dir, subsets=subsets)

    # processes only carried out if the validation subsets are created
    if validation == True:
        # creates directories for validation subsets
        if os.path.exists(os.path.join(save_path, 'MSI_validation_subsets')) == False:
            os.mkdir(os.path.join(save_path, 'MSI_validation_subsets'))
        else:
            check_folder_contents(os.path.join(save_path, 'MSI_validation_subsets'))
        if os.path.exists(os.path.join(save_path, 'MSS_validation_subsets')) == False:
            os.mkdir(os.path.join(save_path, 'MSS_validation_subsets'))
        else:
            check_folder_contents(os.path.join(save_path, 'MSS_validation_subsets'))

        MSI_validation_dir = create_subsets(MSI_dir, 'MSI_validation_subsets', sample_size=int(sample_size*0.8), n_jobs=4)
        MSS_validation_dir = create_subsets(MSS_dir, 'MSS_validation_subsets', sample_size=int(sample_size*0.8), n_jobs=4)

        sort_tiles(MSI_validation_dir, MSS_validation_dir, subsets=subsets)


    if validation == True:
        return MSI_subsets_dir, MSS_subsets_dir, MSI_validation_dir, MSS_validation_dir, save_path
    else:
        return MSI_subsets_dir, MSS_subsets_dir, save_path


def main(MSI_dir, MSS_dir, sample_size=5000, n_jobs=1, cohort_name='', validation=False, subsets=True):
    # creates the subsets for each to run on a seperate CPU core, as well as validation splits from the data
    if validation == True:
        MSI_subsets_dir, MSS_subsets_dir, MSI_validation_dir, MSS_validation_dir, save_path = init_subdirs(MSI_dir, MSS_dir, sample_size=sample_size, n_jobs=n_jobs, validation=validation, subsets=subsets)
    else:
        MSI_subsets_dir, MSS_subsets_dir, save_path = init_subdirs(MSI_dir, MSS_dir, sample_size=sample_size, n_jobs=n_jobs, validation=validation, subsets=subsets)
    
    # GLCM creation and texture analysis performed on tiles
    feature_extraction(MSI_subsets_dir, os.path.join(save_path, 'MSI'))
    feature_extraction(MSS_subsets_dir, os.path.join(save_path, 'MSS'))

    # GLCM creation and texture analysis performed on validation subset tiles
    if validation == True:
        feature_extraction(MSI_validation_dir, os.path.join(save_path, 'MSI_validation_subsets'))
        feature_extraction(MSS_validation_dir, os.path.join(save_path, 'MSS_validation_subsets'))

    # model training and plotting of texture analysis
    X_train, X_test, y_train, y_test = preprocess_data(os.path.join(save_path, 'MSI'), os.path.join(save_path, 'MSS'))
    if validation == True:
        plot_predict(X_train,X_test, y_train, y_test, os.path.join(save_path, 'MSI_validation_subsets'), os.path.join(save_path, 'MSS_validation_subsets'), title=('Prediction of MSI in ' + cohort_name), validation=validation)
    else:
        plot_predict(X_train,X_test, y_train, y_test, title=('Prediction of MSI in ' + cohort_name), validation=validation)

