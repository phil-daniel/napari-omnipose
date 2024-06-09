from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from skimage.morphology import skeletonize
from skimage.io import imsave
from cellpose_omni import models

from napari.utils.notifications import show_warning, show_info
from napari import Viewer

import numpy as np
import csv

if TYPE_CHECKING:
    import napari

import napari

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
    layer_name: str,
    properties: dict,
    show_bounding_boxes: bool,
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
    diameter: int = 25,
) -> "napari.types.LabelsData":
    masks, _, _ = models.CellposeModel(model_type=model).eval(
        [img_data],
        diameter=diameter,
        channels=[1,2],
        omni=True
    )
    show_info(str(np.max(masks[0])) + " objects identified.")
    return masks

def add_labelling(
    viewer: Viewer,
    segmentation_mask,
    cell_count: bool = False,
    bounding_box: bool = False,
    display_area: bool = False,
) -> None:
    info = ['bbox']
    if cell_count:
        info.append('label')
    if display_area:
        info.append('area')
    properties = regionprops_table(
        segmentation_mask,
        properties = tuple(info),
    )
    create_label_layer(viewer, "segmentation label", properties, bounding_box)
    return

def get_properties(
    segmentation_mask,
    bounding_box: bool = False,
    count: bool = False,
    area: bool = False,
    perimeter: bool = False,
) -> dict:
    info = []
    if bounding_box: info.append('bbox')
    if count: info.append('label')
    if area: info.append('area')
    if perimeter: info.append('perimeter')
    properties = regionprops_table(
        segmentation_mask,
        properties = tuple(info),
    )
    return properties

def get_background_intensity(
    segmentation,
    intensity_data,
    min_dist: int,
    max_dist: int,
) -> dict:
    background = np.subtract(expand_labels(segmentation, max_dist), expand_labels(segmentation, min_dist))
    background_intensity = regionprops_table(
        label_image = background,
        intensity_image = intensity_data,
        properties = {'label', 'intensity_mean'}
    )
    background_dict = {}
    for i in range(len(background_intensity['label'])):
        background_dict[background_intensity['label'][i]] = background_intensity['intensity_mean'][i]
    return background_dict

def get_intensity_properties(
    segmentation_mask,
    intensity_data,
    min_dist: int,
    max_dist: int,
    intensity_mean: bool = False,
    intensity_min: bool = False,
    intensity_max: bool = False,
) -> dict:
    info = ['label', 'bbox']
    if intensity_mean: info.append('intensity_mean')
    if intensity_min: info.append('intensity_min')
    if intensity_max: info.append('intensity_max')
    properties = regionprops_table(
        label_image = segmentation_mask,
        intensity_image = intensity_data,
        properties = tuple(info),
    )
    background_intensity = get_background_intensity(
        segmentation = segmentation_mask,
        intensity_data = intensity_data,
        min_dist = min_dist,
        max_dist = max_dist,
    )
    for i in range(len(properties['label'])):
        if background_intensity.get(properties['label'][i]):
            if intensity_mean: properties['intensity_mean'][i] -= background_intensity[properties['label'][i]]
            if intensity_min: properties['intensity_min'][i] -= background_intensity[properties['label'][i]]
            if intensity_max: properties['intensity_max'][i] -= background_intensity[properties['label'][i]]
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
    show_background_areas: bool = True,
    min_dist: int = 5,
    max_dist: int = 10,
) -> None:
    if min_dist >=  max_dist:
        show_warning("Minimum distance is greater or equal to maximum distance")
        return None
    intensity = get_intensity_properties(
        segmentation_mask = seg_layer.data,
        intensity_data = intensity_image.data,
        min_dist = min_dist,
        max_dist = max_dist,
        intensity_mean = True,
    )
    if show_background_areas and viewer != None:
        background = np.subtract(expand_labels(seg_layer.data, max_dist), expand_labels(seg_layer.data, min_dist))
        viewer.add_labels(background, name="Intensity background")
    create_label_layer(viewer, "Intensity Labels", intensity, True)
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
    add_labelling(viewer, seg_layer.data, show_cell_count, show_bounding_box, show_area)
    return None


@magic_factory(
    model = dict(widget_type='ComboBox', label='Model', choices=['bact_phase_omni', 'bact_fluor_omni', 'nuclei', 'cyto', 'cyto2'], value='bact_phase_omni'),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value= False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value=False),
    show_area = dict(widget_type="CheckBox", text="Show Cell Area", value=False),
)
def segment_image(
    viewer: Viewer,
    img_layer: "napari.layers.Image",
    model,
    diameter,
    show_bounding_box,
    show_cell_count,
    show_area,
) -> None:
    if img_layer == None:
        show_warning("No image layer selected.")
        return None
    masks = get_segmentation_mask(img_layer.data, model, diameter)
    viewer.add_labels(masks, name='Segmentation')
    if show_bounding_box or show_cell_count or show_area:
        add_labelling(viewer, masks[0], show_bounding_box, show_cell_count, show_area)
    return

@magic_factory(
    save_directory = dict(widget_type='FileEdit', mode='d', label='Save to directory', value="~/"),
    file_name = dict(value = "ImageAnalysis"),
    model = dict(widget_type='ComboBox', label='Model', choices=['bact_phase_omni', 'bact_fluor_omni', 'nuclei', 'cyto', 'cyto2'], value='bact_phase_omni'),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
)
def full_analysis(
    viewer: Viewer,
    image: "napari.layers.Image",
    intensity_image: "napari.layers.Image",
    save_directory,
    file_name: str,
    model: str,
    diameter: int,
    min_dist: int = 5,
    max_dist: int = 10,
) -> None:
    # Hiding all irrelevant layers to ensure screenshot shows correct information.
    for layer in viewer.layers:
        layer.visible = False
    image.visible = True
    # Adding segmentation layer and label layer
    segmentation_mask = get_segmentation_mask(image.data, model, diameter)
    viewer.add_labels(segmentation_mask, name='Segmentation Mask')
    add_labelling(viewer, segmentation_mask[0], True, True, False)
    properties = get_properties(
        segmentation_mask = segmentation_mask[0],
        count = True,
        area = True,
        perimeter = True,
    )
    intensity_properties = get_intensity_properties(
        segmentation_mask = segmentation_mask[0],
        intensity_data = intensity_image.data,
        min_dist = min_dist,
        max_dist = max_dist,
        intensity_mean = True,
        intensity_min = True,
        intensity_max = True,
    )
    for key in intensity_properties.keys():
        if key not in properties.keys():
            properties[key] = intensity_properties[key]
    row_names = np.array([key for key in properties.keys()])
    prop_data = np.array([properties[key] for key in properties.keys()])
    output = np.append([row_names], np.transpose(prop_data), axis=0)
    # TOTAL INTENSITY



    # Writing output of the image analysis to a csv file
    np.savetxt(str(save_directory)+'/'+file_name+'.csv', output, delimiter=",", fmt='%s')
    # Saving screenshot image of viewer
    imsave(str(save_directory)+'/'+file_name+'.png', viewer.screenshot())
    return