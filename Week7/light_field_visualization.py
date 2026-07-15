import cv2
import numpy as np

def create_light_field():
    """
    Simulates a 4D light field by rendering a 3D object from various viewpoints.
    In a real scenario, a light field is captured by a camera array or a single
    camera moving along a grid.
    """
    # Define a simple 3D scene: a blue square acting as an object
    scene = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a blue object (representing a 3D object in the scene)
    cv2.rectangle(scene, (50, 50), (150, 150), (255, 0, 0), -1)

    light_field = []
    # Simulate viewpoints along a small grid (u, v)
    for v in range(5):  # v (vertical) position
        row = []
        for u in range(5):  # u (horizontal) position
            # Create a copy of the scene
            view = scene.copy()
            # Simulate perspective change by shifting the object
            # This is a highly simplified approximation
            shift_x = (u - 2) * 5
            shift_y = (v - 2) * 5
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            view = cv2.warpAffine(view, M, (view.shape[1], view.shape[0]))
            row.append(view)
        # Vertically stack the row to create the light field display
        light_field.append(np.hstack(row))

    # Vertically stack all rows to form the final light field image
    full_field = np.vstack(light_field)
    return full_field

# Display the light field
light_field_img = create_light_field()
cv2.imshow('Light Field Visualization (Viewpoints)', light_field_img)
print("Light Field Simulation: Each sub-image represents a different viewpoint.")
print("Press any key to close the window.")
cv2.waitKey(0)
cv2.destroyAllWindows()
