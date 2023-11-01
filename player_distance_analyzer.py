import cv2
import numpy as np

def calculate_time(S):
    a = 5
    v1 = 6
    X = S/10
    # Calculate the distance covered during the acceleration phase
    s1 = (v1 ** 2) / (2 * a)

    # Check if the total distance X is less than or equal to the distance covered during the acceleration phase
    mask = (X <= s1).astype(np.uint8)
    t1 = np.sqrt(2 * X / a)

    result = t1*mask

    s2 = X - s1
    t1 = v1 / a
    t2 = s2 / v1
    tsm =  t1 + t2
    result+= (1-mask)*tsm
    return result

def calculate_disc_time(S):
    X = S / 10
    v1 = 4+(X/70)*10

    return X/v1

class RectangleImage:
    def __init__(self, width, height, color=(0, 255, 0)):
        self.width = width
        self.height = height
        self.color = color
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.draw_rectangle()

    def draw_rectangle(self):
        cv2.rectangle(self.image, (0, 0), (self.width, self.height), self.color, -1)

    def show(self, callback):
        cv2.imshow('Rectangle Image', self.image)
        cv2.setMouseCallback('Rectangle Image', callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class PointCollector:
    def __init__(self):
        self.left_points = []
        self.right_points = []

        self.dragging = False
        self.dragged_point = None
        self.dragged_point_type = None
        self.dragged_idx = None

    def on_click(self, event, x, y, flags, param):
        image = param['image']
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_point, point_type, closest_idx, dist = self.find_closest_point(x, y)
            if closest_point is not None and dist < 5:
                # Start dragging the closest point
                self.dragging = True
                self.dragged_point = closest_point
                self.dragged_idx = closest_idx
                self.dragged_point_type = point_type
                temp_image = image.copy()
                radius = 15
                circle_color = (255, 255, 0)
                cv2.circle(temp_image, (x, y), radius, circle_color, -1)

                cv2.imshow('Rectangle Image', temp_image)
            else:
                # Add a new point and draw it
                print(f'Left Clicked at position ({x}, {y})')
                self.left_points.append((x, y))
                radius = 5
                circle_color = (255, 0, 0)  # Blue color
                cv2.circle(image, (x, y), radius, circle_color, -1)
                self.update_background(image)
                cv2.imshow('Rectangle Image', image)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # Update the position of the dragged point and redraw it
                self.dragged_point = (x,y)

                # self.update_background(image)
                temp_image = image.copy()
                radius = 15
                circle_color = (255, 255, 0)
                cv2.circle(temp_image, (x, y), radius, circle_color, -1)

                cv2.imshow('Rectangle Image', temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                # Stop dragging and update the point list
                if self.dragged_point_type == "left":
                    index = self.dragged_idx
                    self.left_points[index] = (x, y)
                elif self.dragged_point_type == "right":
                    index = self.dragged_idx
                    self.right_points[index] = (x, y)
                self.dragging = False
                self.update_background(image)
                cv2.imshow('Rectangle Image', image)
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f'Right Clicked at position ({x}, {y})')
            # Store the right clicked point in the list
            self.right_points.append((x, y))

            # Update the background color
            self.update_background(image)
            # Update the image display
            cv2.imshow('Rectangle Image', image)

    def find_closest_point(self, x, y):
        closest_point = None
        point_type = None
        closest_point_id = None
        min_distance = float('inf')
        for i, point in enumerate(self.left_points + self.right_points):
            distance = np.linalg.norm(np.array(point) - np.array([x, y]))
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                closest_point_id = i if i<len(self.left_points) else i-len(self.left_points)
                point_type = "left" if point in self.left_points else "right"
        return closest_point, point_type,closest_point_id,min_distance

    def update_background(self, image):
        colormap = cv2.COLORMAP_JET
        height, width, _ = image.shape
        y_coords, x_coords = np.indices((height, width))

        if len(self.left_points) > 1:

            red_distances = np.sqrt(
                (x_coords[np.newaxis, :, :] - np.array(self.left_points)[1:, 0][:, np.newaxis, np.newaxis]) ** 2 + (
                            y_coords - np.array(self.left_points)[1:, 1][:, np.newaxis, np.newaxis]) ** 2)

            red_distances = red_distances.min(axis=0)
            red_distances = np.squeeze(red_distances)
            red_distances = calculate_time(red_distances)
        else:
            red_distances = np.zeros((height, width), dtype=float)

        if len(self.right_points) > 0:
            blue_distances = np.sqrt(
                (x_coords[np.newaxis, :, :] - np.array(self.right_points)[:, 0][:, np.newaxis, np.newaxis]) ** 2 + (
                        y_coords - np.array(self.right_points)[:, 1][:, np.newaxis, np.newaxis]) ** 2)

            blue_distances = blue_distances.min(axis=0)
            blue_distances = np.squeeze(blue_distances)
            blue_distances = calculate_time(blue_distances)
        else:
            blue_distances = np.zeros((height, width), dtype=float)

        if len(self.left_points) > 0 and len(self.right_points) > 0:
            handler_distances = np.sqrt(
                (x_coords[np.newaxis, :, :] - np.array([self.left_points[0]])[:, 0][:, np.newaxis, np.newaxis]) ** 2 + (
                        y_coords - np.array([self.left_points[0]])[:, 1][:, np.newaxis, np.newaxis]) ** 2)
            handler_distances = np.squeeze(handler_distances)
            handler_distances = calculate_disc_time(handler_distances)
            disc_mask = np.logical_and(handler_distances > red_distances,handler_distances<blue_distances)

        else:
            disc_mask = None


        # normalized_array = cv2.normalize(red_distances, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # colored_array = cv2.applyColorMap(normalized_array, colormap)
        # cv2.imshow('blue', colored_array)
        # cv2.waitKey(1)
        #
        # normalized_array = cv2.normalize(blue_distances, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # colored_array = cv2.applyColorMap(normalized_array, colormap)
        # cv2.imshow('red', colored_array)
        # cv2.waitKey(1)

        red_mask = (blue_distances > red_distances)
        if disc_mask is not None:
            red_mask *= disc_mask


        blue_mask = np.logical_not(red_mask)

        image[blue_mask] = (255, 170, 170)
        image[red_mask] = (170, 170, 255)


        for i,p in enumerate(self.left_points) :
            radius = 5
            if i >0:
                circle_color = (0, 0, 255)  # Red color
            else:
                circle_color = (128,128,255)
                cv2.circle(image, p, 7, (255,255,255), 1)
            cv2.circle(image, p, radius, circle_color, -1)

        for p in self.right_points:
            radius = 5
            circle_color = (255, 0, 0)  # Red color
            cv2.circle(image, p, radius, circle_color, -1)

if __name__ == '__main__':

    # Create a rectangle image with width=500, height=300, and color (red, green, blue)=(0, 128, 0)
    rect_image = RectangleImage(width=250, height=600, color=(0, 128, 0))

    # Create a PointCollector instance to handle the clicks
    point_collector = PointCollector()

    # Show the image and handle click events using the on_click method of the PointCollector
    rect_image.show(callback=lambda event, x, y, flags, param: point_collector.on_click(event, x, y, flags, param={'image': rect_image.image}))
