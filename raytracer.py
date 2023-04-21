import numpy as np
import matplotlib.pyplot as plt
from light import *

class Ray:
    def __init__(self, ray_origin, ray_direction, ray_properties):
        self.origin = ray_origin
        self.direction = ray_direction
        self.properties = ray_properties

    def ray_intersect_segment(self, segment_A, segment_B):
        """
        Source for equations: https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
        """
        print(segment_A,segment_B)
        v1 = self.origin - segment_A
        v2 = segment_B - segment_A

        if np.cross(v2, self.direction) < 0:
            v3 = (-self.direction[1], self.direction[0])
        else:
            v3 = (self.direction[1], -self.direction[0])

        if np.abs(np.dot(v2,v3)) < 1e-6:
            return None

        t1 = np.abs(np.cross(v2,v1))/np.dot(v2,v3)
        t2 = np.dot(v1,v3) / np.dot(v2,v3)

        print(t1,t2)
        if t1 >= 0 and 0 <= t2 <= 1:
            return self.origin + t1 * self.direction
        else:
            return None


    def intersect_triangle(self, triangle_vertices):
        intersections = [self.ray_intersect_segment(vertex1, vertex2) for (vertex1,vertex2) in [(triangle_vertices[0], triangle_vertices[1]), (triangle_vertices[1], triangle_vertices[2]), (triangle_vertices[2], triangle_vertices[0])]]

        distance_to_intersections = []
        for intersection in intersections:
            if intersection is None:
                distance_to_intersections.append(np.Inf)
            else:
                distance_to_intersections.append(np.sqrt(np.sum(np.square(intersection - self.origin))))
        if np.min(distance_to_intersections) == np.Inf:
            return None
        else:
            return intersections[np.argmin(distance_to_intersections)]

triangle_vertices = np.array([[0, 0], [2, 0], [1, 3]])
light_source = np.array([0.5, 4])

ray_direction = np.array([0, -1])

red_ray = Ray(light_source + np.array([0.1, 0]), ray_direction, red_light)
orange_ray = Ray(light_source + np.array([0.2, 0]), ray_direction, orange_light)
yellow_ray = Ray(light_source + np.array([0.3, 0]), ray_direction, yellow_light)
green_ray = Ray(light_source + np.array([0.4, 0]), ray_direction, green_light)
blue_ray = Ray(light_source + np.array([0.5, 0]), ray_direction, blue_light)
indigo_ray = Ray(light_source + np.array([0.6, 0]), ray_direction, indigo_light)
violet_ray = Ray(light_source + np.array([0.7, 0]), ray_direction, violet_light)

rays = [red_ray, orange_ray, yellow_ray, green_ray, blue_ray, indigo_ray, violet_ray]

for i, ray in enumerate(rays):
    # ray_origin = [light_source[0] + i * 0.1, light_source[1]]
    intersection = ray.intersect_triangle(triangle_vertices)
    plt.plot(ray.origin[0], ray.origin[1], marker='o', color=ray.properties.color)

    if intersection is not None:
        # plt.plot([ray_origin[0], intersection[0]], [ray_origin[1], intersection[1]], color='red')
        plt.plot([ray.origin[0],intersection[0]],[ray.origin[1],intersection[1]], color=ray.properties.color)
        # plt.plot(intersection[0], intersection[1], marker='x', color='red')

plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color='black')
plt.plot([triangle_vertices[0, 0], triangle_vertices[-1, 0]], [triangle_vertices[0, 1], triangle_vertices[-1, 1]], color='black')
plt.axis('equal')
plt.show()

