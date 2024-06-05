from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels


from napari.utils.notifications import show_warning, show_info
from napari import Viewer

import numpy as np

if TYPE_CHECKING:
    import napari

import napari

def make_bounding_box(coords):
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
    print(box)
    box = np.moveaxis(box, 2, 0)
    print(box)
    return box

@magic_factory(
)
def measure_masks(
    mask: "napari.layers.Labels",
) -> None:
    properties = regionprops_table(
        mask.data,
        properties = {'bbox'}
    )
    boxes = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
    for i, x in enumerate(boxes):
        items = np.where(mask == i+1)
        # looking for i+1
    
    return


def add_labelling(
    viewer: Viewer,
    segmentation_mask,
    bounding_box: bool,
    cell_count: bool,
    display_area: bool,
) -> None:
    properties = regionprops_table(
        segmentation_mask,
        properties = ('label', 'bbox', 'perimeter', 'area'),
    )
    boxes = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
    labelText = []
    if cell_count:
        labelText.append("{label}")
    if display_area:
        labelText.append("Area: {area}")
    viewer.add_shapes(
        boxes,
        shape_type = 'rectangle',
        face_color = 'transparent',
        edge_color = 'yellow',
        edge_width = 2 if bounding_box else 0,
        properties = properties,
        text = {
            'string': "\n".join(labelText),
            'size': 10,
            'color': 'yellow',
            'anchor': 'upper_left',
            'translation': [-3, 0],
        },
        name='Segmentation Labelling',
    )
    return

@magic_factory(
)
def calculate_intensity(
    seg_layer: "napari.layers.Labels",
    img_layer: "napari.layers.Image",
    min_dist: int = 5,
    max_dist: int = 10,
) -> napari.layers.Labels:
    # todo
    # subtract background from indiviudal calc
    if min_dist >=  max_dist:
        show_warning("Minimum distance is greater or equal to maximum distance")
        return None
    cell_intensity = regionprops_table(
        label_image = seg_layer.data,
        intensity_image = img_layer.data,
        properties = {'label', 'area', 'intensity_mean'}
    )
    background = np.subtract(expand_labels(seg_layer.data, max_dist), expand_labels(seg_layer.data, min_dist))
    background_intensity = regionprops_table(
        label_image = background,
        intensity_image = img_layer.data,
        properties = {'label', 'intensity_mean'}
    )
    background_dict = {}
    for i in range(len(background_intensity['label'])):
        background_dict[background_intensity['label'][i]] = background_intensity['intensity_mean'][i]
    for i in range(len(cell_intensity['label'])):
        if background_dict.get(cell_intensity['label'][i]):
            cell_intensity['intensity_mean'][i] -= background_dict[cell_intensity['label'][i]]
    print (cell_intensity)
    return napari.layers.Labels(data = background, name = "Background intensity areas")

@magic_factory(
)
def remove_segmented_object(
    seg_layer: "napari.layers.Labels",
    int_value: int
) -> "napari.layers.Labels":
    if seg_layer == None:
        show_warning("No label layer selected.")
        return None
    newData = np.copy(seg_layer.data)
    newData[newData == int_value] = 0
    if np.max(newData) > int_value:
        newData[newData == np.max(newData)] = int_value
    seg_layer.visible = False
    return napari.layers.Labels(newData)

@magic_factory(
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value= False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value= False),
    show_area = dict(widget_type="CheckBox", text="Show Cell Area", value=False),
)
def label_segmentation(
    seg_layer: "napari.layers.Labels",
    show_bounding_box,
    show_cell_count,
    show_area,
    viewer: Viewer,
) -> "None":
    if seg_layer == None:
        show_warning("No label layer selected.")
        return None
    add_labelling(viewer, seg_layer.data, show_bounding_box, show_cell_count, show_area)
    return None


@magic_factory(
    model = dict(widget_type='ComboBox', label='Model', choices=['bact_phase_omni', 'bact_fluor_omni', 'nuclei', 'cyto', 'cyto2'], value='bact_phase_omni'),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value= False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value=False),
    show_area = dict(widget_type="CheckBox", text="Show Cell Area", value=False),
)
def segment_image(
    img_layer: "napari.layers.Image",
    model,
    diameter,
    show_bounding_box,
    show_cell_count,
    show_area,
    viewer: Viewer,
) -> "napari.types.LabelsData":
    from cellpose_omni import models
    if img_layer == None:
        show_warning("No image layer selected.")
        return None
    img = img_layer.data
    masks, flows, styles = models.CellposeModel(model_type=model).eval([img],
                            diameter=diameter,
                            channels=[1,2],
                            omni=True)
    show_info(str(np.max(masks[0])) + " objects identified.")
    if show_bounding_box or show_cell_count:
        add_labelling(viewer, masks[0], show_bounding_box, show_cell_count, show_area)

    return masks