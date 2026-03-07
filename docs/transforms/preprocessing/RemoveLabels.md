# RemoveLabels

::: torchio.transforms.RemoveLabels

```python plot
import torchio as tio
colin = tio.datasets.Colin27(2008)
label_map = colin.cls
colin.remove_image('t2')
colin.remove_image('pd')
names_to_remove = (
    'Fat',
    'Muscles',
    'Skin and Muscles',
    'Skull',
    'Fat 2',
    'Dura',
    'Marrow'
)
labels = [colin.NAME_TO_LABEL[name] for name in names_to_remove]
skull_stripping = tio.RemoveLabels(labels)
only_brain = skull_stripping(label_map)
colin.add_image(only_brain, 'brain')
colors = {
    0: (0, 0, 0),
    1: (127, 255, 212),
    2: (96, 204, 96),
    3: (240, 230, 140),
    4: (176, 48, 96),
    5: (48, 176, 96),
    6: (220, 247, 164),
    7: (103, 255, 255),
    9: (205, 62, 78),
    10: (238, 186, 243),
    11: (119, 159, 176),
    12: (220, 216, 20),
}
colin.plot(cmap_dict={'cls': colors, 'brain': colors})
```
