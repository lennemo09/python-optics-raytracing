import numpy as np
import matplotlib.pyplot as plt

def ray_intersect_segment(ray_origin, ray_direction, segment_A, segment_B):
    """
    Source for equations: https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
    """
    print(segment_A,segment_B)
    v1 = ray_origin - segment_A
    v2 = segment_B - segment_A
    v3 = (-ray_direction[1], ray_direction[0])

    if np.abs(np.dot(v2,v3)) < 1e-6:
        return None

    t1 = np.cross(v2,v1)/np.dot(v2,v3)
    t2 = np.dot(v1,v3) / np.dot(v2,v3)

    print(t1,t2)
    if t1 >= 0 and 0 <= t2 <= 1:
        return ray_origin + t1 * ray_direction
    else:
        return None

def intersect_triangle(ray_origin, ray_direction, triangle_vertices):
    intersections = [ray_intersect_segment(ray_origin, ray_direction, vertex1, vertex2) for (vertex1,vertex2) in [(triangle_vertices[0], triangle_vertices[1]), (triangle_vertices[1], triangle_vertices[2]), (triangle_vertices[2], triangle_vertices[0])]]

    distance_to_intersections = []
    for intersection in intersections:
        if intersection is None:
            distance_to_intersections.append(np.Inf)
        else:
            distance_to_intersections.append(np.sqrt(np.sum(np.square(intersection - ray_origin))))
    if np.min(distance_to_intersections) == np.Inf:
        return None
    else:
        return intersections[np.argmin(distance_to_intersections)]

triangle_vertices = np.array([[0, 0], [1, 3], [2, 0] ])
light_source = np.array([3, 2])

ray_direction = np.array([-1, 0])

for i in range(1):
    ray_origin = light_source
    intersection = intersect_triangle(ray_origin, ray_direction, triangle_vertices)
    print(intersection)
    if intersection is not None:
        # plt.plot([ray_origin[0], intersection[0]], [ray_origin[1], intersection[1]], color='red')
        plt.plot(intersection[0], intersection[1], marker='x', color='red')

plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color='black')
plt.plot([triangle_vertices[0, 0], triangle_vertices[-1, 0]], [triangle_vertices[0, 1], triangle_vertices[-1, 1]], color='black')
plt.plot(light_source[0], light_source[1], marker='o', color='blue')
plt.axis('equal')
plt.show()

