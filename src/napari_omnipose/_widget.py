from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.measure import regionprops_table


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
    box = np.moveaxis(box, 2, 0)
    return box

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
    print (models)
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