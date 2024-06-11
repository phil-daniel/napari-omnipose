from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from skimage.io import imsave
from cellpose_omni import models

from napari.utils.notifications import show_warning, show_info
from napari import Viewer

import numpy as np

if TYPE_CHECKING:
    import napari

import napari

# extra region props
def intensity_std(
    segmentation_mask: np.ndarray,
    intensity_data: np.ndarray,
) -> np.ndarray:
    return np.std(intensity_data[segmentation_mask])

def make_bounding_box(
    coords,
):
    minr = coords[0]
    minc = coords[1]
    maxr = coords[2]
    maxc = coords[3]

    box = np.array(
        [
            [minr, minc],
            [maxr, minc],
            [maxr, maxc],
            [minr, maxc]
        ]
    )
    box = np.moveaxis(box, 2, 0)
    return box


# Adds a new label layer to the view, consisting of all the information in properties
def create_label_layer(
    viewer: Viewer,
    properties: dict,
    layer_name: str = "New Layer",
    show_bounding_boxes: bool = False,
) -> None:
    boxes = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
    labelText = ["{label}"] if 'label' in properties else []
    props_to_ignore = {'label', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3'}
    for prop in properties.keys():
        if prop not in props_to_ignore:
            labelText.append(prop.capitalize() + ": {" + prop + "}")
    viewer.add_shapes(
        boxes,
        shape_type = 'rectangle',
        face_color = 'transparent',
        edge_color = 'yellow',
        edge_width = 2 if show_bounding_boxes else 0,
        properties = properties,
        text = {
            'string': "\n".join(labelText),
            'size': 10,
            'color': 'yellow',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        },
        name = layer_name,
    )
    return

# Returns a mask of the image using the cellpose model inputted
def get_segmentation_mask(
    img_data: "napari.types.ImageData",
    model: str = 'bact_phase_omni',
    custom_model = None,
    diameter: int = 25,
    use_gpu: bool = False
) -> "napari.types.LabelsData":

    if custom_model != None:
        model = models.CellposeModel(
            gpu = use_gpu,
            pretrained_model = str(custom_model),
            nchan = 2,
            nclasses = 3,
            dim = 2
        )
    else:
        model = models.CellposeModel(
            gpu=use_gpu,
            model_type=model
        )
    masks, _, _ = model.eval(
        [img_data],
        diameter=diameter,
        channels=[0,0],
        omni=True,
    )
    show_info(str(np.max(masks[0])) + " objects identified.")
    return masks

# Returns a dictionary containing the information below
def get_properties(
    segmentation_mask: np.ndarray,
    bounding_box: bool = False,
    count: bool = False,
    area: bool = False,
    perimeter: bool = False,
    centroid: bool = False,
) -> dict:
    info = []
    if bounding_box: info.append('bbox')
    if count: info.append('label')
    if area: info.append('area')
    if perimeter: info.append('perimeter')
    if centroid: info.append('centroid')
    properties = regionprops_table(
        segmentation_mask,
        properties = tuple(info),
    )
    return properties

def add_labelling(
    viewer: Viewer,
    segmentation_mask: np.ndarray,
    cell_count: bool = False,
    bounding_box: bool = False,
    display_area: bool = False,
) -> None:
    properties = get_properties(
        segmentation_mask = segmentation_mask,
        bounding_box = bounding_box,
        count = cell_count,
        area = display_area,
    )
    create_label_layer(
        viewer = viewer,
        properties = properties,
        layer_name = "Segmentation Labels",
        show_bounding_boxes = bounding_box,
    )
    return

def get_background_intensity(
    segmentation: np.ndarray,
    intensity_data: np.ndarray, 
    expansion_dist: int = 10,
) -> int:
    background = expand_labels(segmentation, expansion_dist)
    background[background == 0] = np.max(background)+1
    background[background != np.max(background)] = 0
    background[background == np.max(background)] = 1
    background_intensity = regionprops_table(
        label_image = background,
        intensity_image = intensity_data,
        properties = {'intensity_mean'}
    )
    return background_intensity['intensity_mean'][0]

def get_intensity_properties(
    segmentation_mask,
    intensity_data,
    expansion_dist: int,
    bbox: bool = True,
    intensity_mean: bool = False,
    intensity_min: bool = False,
    intensity_max: bool = False,
    show_intensity_std: bool = False,
    background_mean: bool = False,
    total_intensity: bool = False,
) -> dict:
    info, extra_props = ['label'], []
    if bbox: info.append('bbox')
    if intensity_mean or total_intensity: info.append('intensity_mean')
    if intensity_min: info.append('intensity_min')
    if intensity_max: info.append('intensity_max')
    if show_intensity_std: extra_props.append(intensity_std)
    if total_intensity: info.append('area')
    properties = regionprops_table(
        label_image = segmentation_mask,
        intensity_image = intensity_data,
        properties = tuple(info),
        extra_properties = extra_props
    )
    background_intensity = get_background_intensity(
        segmentation = segmentation_mask,
        intensity_data = intensity_data,
        expansion_dist = expansion_dist,
    )
    if intensity_mean:
        properties['intensity_mean'] = np.subtract(properties['intensity_mean'], background_intensity)
    if intensity_min:
        properties['intensity_min'] = np.subtract(properties['intensity_min'], background_intensity)
    if intensity_max:
        properties['intensity_max'] = np.subtract(properties['intensity_max'], background_intensity)
    if background_mean:
        properties['background_intensity_mean'] = np.full_like(properties['label'], background_intensity)
    if total_intensity:
        properties['total_intensity'] = np.multiply(properties['area'], properties['intensity_mean'])
        properties.pop('area')
        if not intensity_mean: properties.pop('intensity_mean')
    return properties

@magic_factory(
)
def measure_masks(
    mask: "napari.layers.Labels",
) -> "napari.layers.Labels":
    '''properties = regionprops_table(
        mask.data,
        properties = {'bbox'}
    )
    boxes = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
    for i, x in enumerate(boxes):
        items = np.where(mask == i+1)
        # looking for i+1'''
    shrink = expand_labels(mask.data, -1)
    return napari.layers.Labels(shrink)

@magic_factory(
)
def calculate_intensity(
    viewer: Viewer,
    seg_layer: "napari.layers.Labels",
    intensity_image: "napari.layers.Image",
    expansion_dist: int = 10,
) -> None:
    intensity = get_intensity_properties(
        segmentation_mask = seg_layer.data,
        intensity_data = intensity_image.data,
        expansion_dist = expansion_dist,
        intensity_mean = True,
    )
    create_label_layer(
        viewer = viewer,
        properties = intensity,
        layer_name = "Intensity Labels",
        show_bounding_boxes = True
    )
    return

@magic_factory(
)
def remove_segmented_object(
    seg_layer: "napari.layers.Labels",
    int_value: list[int],
) -> None:
    if seg_layer == None:
        show_warning("No label layer selected.")
        return None
    newData = np.copy(seg_layer.data)
    int_value.sort(reverse = True)
    for val in int_value:
        newData[newData == val] = 0
        if np.max(newData) != val:
            newData[newData == np.max(newData)] = val
    seg_layer.data = newData
    return

@magic_factory(
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value=False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value=False),
    show_area = dict(widget_type="CheckBox", text="Show Cell Area", value=False),
)
def label_segmentation(
    viewer: Viewer,
    seg_layer: "napari.layers.Labels",
    show_bounding_box: bool = False,
    show_cell_count: bool = False,
    show_area: bool = False,
) -> None:
    if seg_layer == None:
        show_warning("No label layer selected.")
        return None
    add_labelling(
        viewer = viewer, 
        segmentation_mask = seg_layer.data,
        cell_count = show_cell_count, 
        bounding_box = show_bounding_box,
        display_area = show_area
    )
    return None


@magic_factory(
    use_gpu = dict(widget_type="CheckBox", label = 'Use GPU', value=False),
    model = dict(widget_type='ComboBox', label='Model', choices=['bact_phase_omni', 'bact_fluor_omni', 'nuclei', 'cyto', 'cyto2'], value='bact_phase_omni'),
    custom_model = dict(widget_type='FileEdit', mode='r', label='Custom Model', filter=None, value=None),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value= False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value=False),
    show_area = dict(widget_type="CheckBox", text="Show Cell Area", value=False),
)
def segment_image(
    viewer: Viewer,
    img_layer: "napari.layers.Image",
    use_gpu: bool,
    model: str,
    custom_model = None,
    diameter: int = 25,
    show_bounding_box: bool = False,
    show_cell_count: bool = False,
    show_area: bool = False,
) -> None:
    if img_layer == None:
        show_warning("No image layer selected.")
        return None
    masks = get_segmentation_mask(
        img_data = img_layer.data,
        model = model,
        custom_model = custom_model,
        diameter = diameter,
        use_gpu = use_gpu)
    viewer.add_labels(masks, name='Segmentation')
    if show_bounding_box or show_cell_count or show_area:
        add_labelling(
            viewer = viewer,
            segmentation_mask = masks[0],
            cell_count = show_cell_count,
            bounding_box = show_bounding_box,
            display_area = show_area
        )
    return

@magic_factory(
    save_directory = dict(widget_type='FileEdit', mode='d', label='Save to Directory', value="~/"),
    file_name = dict(value="ImageAnalysis"),
    model = dict(widget_type='ComboBox', label='Model', choices=['bact_phase_omni', 'bact_fluor_omni', 'nuclei', 'cyto', 'cyto2'], value='bact_phase_omni'),
    custom_model = dict(widget_type='FileEdit', mode='r', label='Custom Model', filter=None, value=None),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
    expansion_dist = dict(label="Intensity cell expansion"),
    use_gpu = dict(label="Use GPU for segmentation"),
)
def full_analysis(
    viewer: Viewer,
    image: "napari.layers.Image",
    intensity_image: "napari.layers.Image",
    save_directory,
    file_name: str = "ImageAnalysis",
    use_gpu: bool = False,
    model: str = "bact_phase_omni",
    custom_model = None,
    existing_segmentation: "napari.layers.Labels" = None,
    diameter: int = 25,
    expansion_dist: int = 10,
    show_area: bool = True,
    show_perimeter: bool = True,
    show_centroid: bool = True,
    show_intensity_mean: bool = True,
    show_intensity_min: bool = True,
    show_intensity_max: bool = True,
    show_intensity_std: bool = True,
    show_total_intensity: bool = True,
    show_background_intensity_mean: bool = True,
) -> None:
    # Hiding all irrelevant layers to ensure screenshot shows correct information.
    for layer in viewer.layers:
        layer.visible = False
    if existing_segmentation:
        existing_segmentation.visible = True
    image.visible = True
    # Adding segmentation layer and label layer
    if existing_segmentation: segmentation_mask = [existing_segmentation.data]
    else:
        segmentation_mask = get_segmentation_mask(
            img_data = image.data,
            model = model,
            custom_model = custom_model,
            diameter = diameter,
            use_gpu = use_gpu
        )
        viewer.add_labels(segmentation_mask, name='Segmentation Mask')
    add_labelling(
        viewer = viewer,
        segmentation_mask = segmentation_mask[0],
        cell_count = True,
        bounding_box = True
    )
    properties = get_properties(
        segmentation_mask = segmentation_mask[0],
        count = True,
        area = show_area,
        perimeter = show_perimeter,
        centroid = show_centroid,
    )
    intensity_properties = get_intensity_properties(
        segmentation_mask = segmentation_mask[0],
        intensity_data = intensity_image.data,
        expansion_dist = expansion_dist,
        bbox = False, 
        intensity_mean = show_intensity_mean,
        intensity_min = show_intensity_min,
        intensity_max = show_intensity_max,
        show_intensity_std = show_intensity_std,
        background_mean = show_background_intensity_mean,
        total_intensity = show_total_intensity,
    )
    for key in intensity_properties.keys():
        if key not in properties.keys():
            properties[key] = intensity_properties[key]
    row_names = np.array([key for key in properties.keys()])
    prop_data = np.array([properties[key] for key in properties.keys()])
    output = np.append([row_names], np.transpose(prop_data), axis=0)
    # Writing output of the image analysis to a csv file
    np.savetxt(str(save_directory)+'/'+file_name+'.csv', output, delimiter=",", fmt='%s')
    # Saving screenshot image of viewer
    imsave(str(save_directory)+'/'+file_name+'.png', viewer.screenshot())
    return