from Segmentation import process_image

process_image(
    image_path=r'CRAFT/data/Motion in a 2d plane_page-0002.jpg',
    trained_model='CRAFT\craft_mlt_25k.pth',
    text_threshold=0.7,
    low_text=0.4,
    link_threshold=0.4,
    cuda=True,
    canvas_size=1280,
    mag_ratio=1.5,
    poly=False,
    show_time=False,
    refine=False,
    refiner_model='weights/craft_refiner_CTW1500.pth',
    output_dir='./result/'
)