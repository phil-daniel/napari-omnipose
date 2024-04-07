from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
from skimage.measure import label, regionprops_table


from napari.utils.notifications import show_warning, show_info
from napari import Viewer

import numpy as np

if TYPE_CHECKING:
    import napari

def make_bounding_box(coords):
    minr = coords[0]
    minc = coords[1]
    maxr = coords[2]
    maxc = coords[3]

    box = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    box = np.moveaxis(box, 2, 0)
    return box

def adding_bounding_boxes(viewer: Viewer, segmentation_mask) -> None:
    label_image = segmentation_mask

    properties = regionprops_table(
        label_image, properties = ('label', 'bbox', 'perimeter', 'area')
    )
    boxes = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
    viewer.add_shapes(
        boxes,
        face_color='transparent',
        edge_color='yellow',
        edge_width=2,
        properties=properties,
        #name='Bounding boxes',
    )

@magic_factory(
    clear_layer = dict(widget_type="CheckBox", text="Clear layer first", value=False), 
)
def bounding_box_widget(
    seg_layer: "napari.layers.Labels",
    clear_layer,
    viewer: Viewer
) -> "None":
    if seg_layer == None:
        show_warning("No label layer selected.")
        return None
    if clear_layer:
        viewer.layers["Labels"].data = []
    adding_bounding_boxes(viewer, seg_layer.data)
    return None


@magic_factory(
    model = dict(widget_type='ComboBox', label='Model', choices=['nuclei', 'cyto', 'cyto2', 'cyto3'], value='nuclei'),
    diameter = dict(widget_type="IntSlider", label="Diameter", value="25", min=0, max=100),
    show_bounding_box = dict(widget_type="CheckBox", text="Show bounding boxes", value= False),
    show_cell_count = dict(widget_type="CheckBox", text="Show Cell Count", value=False),
)
def segment_image(
    img_layer: "napari.layers.Image",
    model,
    diameter,
    show_bounding_box,
    show_cell_count,
    viewer: Viewer
) -> "napari.types.LabelsData":
    from cellpose import models
    print (viewer.layers)
    if img_layer == None:
        show_warning("No image layer selected.")
        return None
    img = img_layer.data
    masks, flows, styles = models.CellposeModel(model_type=model).eval(img,
                            diameter=diameter, channels=[1,2])
    # Gives number of cells identified
    print(str(np.max(masks))  + " objects identified.")
    show_info(str(np.max(masks)) + " objects identified.")

    #if show_cell_count:
    #    count_layer = viewer.

    if show_bounding_box:
        adding_bounding_boxes(viewer, masks)

    return masks