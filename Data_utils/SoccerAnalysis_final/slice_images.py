from sahi.slicing import slice_coco

datasets = [
    r"D:\Datasets\SoccerAnalysis\Player Detection.v3i.coco",
    r"D:\Datasets\SoccerAnalysis\VA_Project.v2i.coco",
]

output_folders = [
    r"D:\Datasets\SoccerAnalysis\v3_sahi",
    r"D:\Datasets\SoccerAnalysis\v2_temp_sahi"
]

for dataset, output_folder in zip(datasets, output_folders):
    for folder in ['train', 'valid', 'test']:
        for size in [160, 320, 640]:
            print(f"Processing {dataset}/{folder} with size {size}")
            slice_coco(
                coco_annotation_file_path=fr"{dataset}/{folder}/_annotations.coco.json",
                image_dir=fr"{dataset}/{folder}/",
                output_coco_annotation_file_name="annotations",
                output_dir=fr"{output_folder}_{size}/{folder}",
                slice_width=size,
                slice_height=size,
            )
