import os
import shutil

# seperates the tiles into individual folders based off of cohort sample ID
# supported cohorts are TCGA and CPTAC
def move_tiles(input_dir, study='TCGA'):
    for file in os.listdir(input_dir):
        substring = study
        sample_name_indices = file.find(substring)
        if study == 'TCGA':
            sample_name =  file[sample_name_indices:sample_name_indices+12]
        elif study == 'CPTAC':
            sample_name = file[0:7]
        else:
            sample_name=file

        if os.path.exists(os.path.join(input_dir, sample_name)) == False:
            os.mkdir(os.path.join(input_dir, sample_name))
        shutil.move(os.path.join(input_dir, file), os.path.join(input_dir, sample_name))


# # checks for duplicate samples between MSI and MSS datasets
# def check_samples(MSI_dir, MSS_dir):
#     MSI, MSS = [], []
#     for file in os.listdir(MSI_dir):
#         substring = 'TCGA'
#         sample_name_indices = file.find(substring)
#         sample_name =  file[sample_name_indices:sample_name_indices+12]
#         # sample_name = file[0:7]
#         if sample_name not in MSI:
#             MSI.append(sample_name)
#     for file in os.listdir(MSS_dir):
#         if sample_name not in MSS:
#             MSS.append(sample_name)
#     for file in MSI:
#         if file in MSS:
#             print(file)

#     return MSI, MSS


def sort_tiles(MSI_dir, MSS_dir, subsets=True, study='TCGA'):
    if subsets == True:
        for subset, subset2 in zip(os.listdir(MSI_dir), os.listdir(MSS_dir)):
            move_tiles(os.path.join(MSI_dir, subset), study=study)
            move_tiles(os.path.join(MSS_dir, subset2), study=study)
    else:
        move_tiles(MSI_dir, study=study)
        move_tiles(MSS_dir, study=study)
