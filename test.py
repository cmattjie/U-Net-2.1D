from distutils.command.build_scripts import first_line_re
import os

main_dir = '/A/motomed/datasets/processed/hmd'
imgs = sorted(os.listdir(os.path.join(main_dir, 'CT')))
masks = sorted(os.listdir(os.path.join(main_dir, 'mask')))
delta_slice = 1

subset = ['0001', '0002']

patients_list = list()
for img in imgs:
    if img[:4] not in patients_list:
        patients_list.append(img[:4])

patients_count = {patient : [] for patient in patients_list}
img_slices = list()
mask_slices = list()

# get list of number of slices for each patient
for img in imgs:
    patients_count[img[:4]].append(img)

#filterByPatient = lambda keys: {x: patients_count[x] for x in keys}
#patients_count = filterByPatient(subset)

#img_slices = lambda keys: {x: img_slices[x] for x in keys}
#img_slices = filterByPatient(subset)

#mask_slices = lambda keys: {x: mask_slices[x] for x in keys}
#mask_slices = filterByPatient(subset)

for patient in patients_count:
    first = True
    if patient in subset:
        total = len(patients_count[patient])
        for slice in range(total):
            if slice < delta_slice:
                group = patients_count[patient][slice-slice : slice+delta_slice+1]
                for _ in range(delta_slice-slice):
                    group.insert(0, group[0])
            
            elif slice > total - delta_slice - 1:
                group = patients_count[patient][slice-delta_slice : slice+delta_slice+1]
                for _ in range(delta_slice*2+1-len(group)):
                    group.append(group[-1])

            else:
                group = patients_count[patient][slice-delta_slice : slice+delta_slice+1]

            print(group)
            img_slices.append(group)
            mask_slices.append(patients_count[patient][slice])

#for i in img_slices:
#    print(len(i))