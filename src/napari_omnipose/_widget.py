from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
from skimage.measure import label, regionprops_table


from napari.utils.notifications import show_warning, show_info
from napari import view_image

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
) -> "napari.types.LabelsData":
    from cellpose import models

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
        viewer = view_image()
        # create the properties dictionary
        properties = regionprops_table(
            label_image, properties=('label', 'bbox', 'perimeter', 'area')
        )
        bbox_rects = make_bounding_box([properties[f'bbox-{i}'] for i in range(4)])
        shapes_layer = viewer.add_shapes(
            bbox_rects,
            face_color='transparent',
            edge_color='green',
            properties=properties,
            text=text_parameters,
            name='bounding box',
        )


    return masks