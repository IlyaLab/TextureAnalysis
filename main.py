from subsets import *
from FeatureExtraction import *
from sort_tiles import *
from ML import *

feature_skip = False

def init_subdirs(MSI_dir, MSS_dir, cohort_name = '', sample_size=5000, n_jobs=1, validation=True, subsets=True, study=''):
    # creates the save path for the results of the texture analysis - found in same directory as code
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Texture_Features',(cohort_name + '_texture_analysis_results'))) == False:
        # right_squid
        os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Texture_Features',(cohort_name + '_texture_analysis_results')))
    else:
        # wrong_squid
        global feature_skip 
        feature_skip = True
        # pass
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Texture_Features',(cohort_name + '_texture_analysis_results'))

    # creates directories for MSI and MSS texture results
    if os.path.exists(os.path.join(save_path, 'MSI')) == False:
        os.mkdir(os.path.join(save_path, 'MSI'))

    if os.path.exists(os.path.join(save_path, 'MSS')) == False:
        os.mkdir(os.path.join(save_path, 'MSS'))

    # checks if subset directories already exist
    if (os.path.isdir(os.path.join(MSI_dir, '../MSI_subsets')) == True) and (os.path.isdir(os.path.join(MSS_dir, '../MSS_subsets')) == True):
        # right_squid
        MSI_subsets_dir = os.path.join(MSI_dir, '../MSI_subsets')
        MSS_subsets_dir = os.path.join(MSS_dir, '../MSS_subsets')
        pass
    else:
        # wrong_squid
        # creates the directories to be used by the subsets in multiprocessing - found in same directory as tiles
        MSI_subsets_dir = create_subsets(MSI_dir, 'MSI_subsets', sample_size=sample_size, n_jobs=n_jobs)
        MSS_subsets_dir = create_subsets(MSS_dir, 'MSS_subsets', sample_size=sample_size, n_jobs=n_jobs)

    # sorts tiles into their TCGA labelled samples
    sort_tiles(MSI_subsets_dir,  MSS_subsets_dir, subsets=subsets, study=study)

    # processes only carried out if the validation subsets are created
    if validation == True:
        # creates directories for validation subsets
        if os.path.exists(os.path.join(save_path, 'MSI_validation_subsets')) == False:
            os.mkdir(os.path.join(save_path, 'MSI_validation_subsets'))

        if os.path.exists(os.path.join(save_path, 'MSS_validation_subsets')) == False:
            os.mkdir(os.path.join(save_path, 'MSS_validation_subsets'))

        # checks if validation subsets already exist
        if (os.path.isdir(os.path.join(MSI_dir, '../MSI_validation_subsets')) == True) and (os.path.isdir(os.path.join(MSS_dir, '../MSS_validation_subsets')) == True):
            MSI_validation_dir = os.path.join(MSI_dir, '../MSI_validation_subsets')
            MSS_validation_dir = os.path.join(MSS_dir, '../MSS_validation_subsets')
            pass
        else:
            MSI_validation_dir = create_subsets(MSI_dir, 'MSI_validation_subsets', sample_size=int(sample_size*0.8), n_jobs=4)
            MSS_validation_dir = create_subsets(MSS_dir, 'MSS_validation_subsets', sample_size=int(sample_size*0.8), n_jobs=4)

        sort_tiles(MSI_validation_dir, MSS_validation_dir, subsets=subsets, study=study)

    if validation == True:
        return MSI_subsets_dir, MSS_subsets_dir, MSI_validation_dir, MSS_validation_dir, save_path
    else:
        return MSI_subsets_dir, MSS_subsets_dir, save_path


def main(MSI_dir, MSS_dir, sample_size=5000, n_jobs=1, cohort_name='', validation=False, subsets=True, study='', model='', ML='XGBoost'):
    
    n_jobs = n_jobs_count(n_jobs=n_jobs)
    
    # squid

    # creates the subsets for each to run on a seperate CPU core, as well as validation splits from the data
    if validation == True:
        MSI_subsets_dir, MSS_subsets_dir, MSI_validation_dir, MSS_validation_dir, save_path = init_subdirs(MSI_dir, MSS_dir, 
                                                                                                           cohort_name=cohort_name, 
                                                                                                           sample_size=sample_size, n_jobs=n_jobs, 
                                                                                                           validation=validation, subsets=subsets, 
                                                                                                           study=study)
    else:
        MSI_subsets_dir, MSS_subsets_dir, save_path = init_subdirs(MSI_dir, MSS_dir, 
                                                                   cohort_name=cohort_name, 
                                                                   sample_size=sample_size, n_jobs=n_jobs, 
                                                                   validation=validation, subsets=subsets, 
                                                                   study=study)
    
    print('Tile sorting complete.')
    if feature_skip == False:
        # GLCM creation and texture analysis performed on tiles
        feature_extraction(MSI_subsets_dir, os.path.join(save_path, 'MSI'))
        feature_extraction(MSS_subsets_dir, os.path.join(save_path, 'MSS'))

        # GLCM creation and texture analysis performed on validation subset tiles
        if validation == True:
            feature_extraction(MSI_validation_dir, os.path.join(save_path, 'MSI_validation_subsets'))
            feature_extraction(MSS_validation_dir, os.path.join(save_path, 'MSS_validation_subsets'))

    print('Feature extraction complete.')

    # model training and plotting of texture analysis
    X_train, X_test, y_train, y_test, data, test_ind = preprocess_data(os.path.join(save_path, 'MSI'), os.path.join(save_path, 'MSS'))
    if validation == True:
        y_pred = plot_predict(X_train,X_test, y_train, y_test, data, 
                              os.path.join(save_path, 'MSI_validation_subsets'), 
                              os.path.join(save_path, 'MSS_validation_subsets'), 
                              title=('Prediction of MSI in ' + cohort_name), 
                              validation=validation, model=model, ML=ML)
    else:
        y_pred = plot_predict(X_train,X_test, y_train, y_test, data, 
                              title=('Prediction of MSI in ' + cohort_name), 
                              validation=validation, model=model, ML=ML)
    
    per_patient(y_pred, y_test, test_ind, 
                title=('Per Patient Prediction of MSI in ' + cohort_name), study=study)

    print('Machine learning complete.')
