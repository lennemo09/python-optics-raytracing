import numpy as np
import matplotlib.pyplot as plt
from math import hypot
from light import *

MAX_BOUNCE = 10

class Ray:
    def __init__(self, ray_origin, ray_direction, ray_properties, medium):
        self.origin = ray_origin
        self.direction = ray_direction
        self.properties = ray_properties
        self.medium = medium

    def ray_intersect_segment(self, segment_A, segment_B):
        """
        Source for equations: https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
        """
        v1 = self.origin - segment_A
        v2 = segment_B - segment_A

        normal1 = np.array([v2[1],-v2[0]])
        normal2 = np.array([-v2[1],v2[0]])

        if np.cross(v2, self.direction) < 0:
            v3 = (-self.direction[1], self.direction[0])
        else:
            v3 = (self.direction[1], -self.direction[0])

        if np.abs(np.dot(v2,v3)) < 1e-6:
            return [None, None, None]

        t1 = np.abs(np.cross(v2,v1)) / np.dot(v2,v3)
        t2 = np.dot(v1,v3) / np.dot(v2,v3)

        if t1 > 1e-6 and 1e-6 <= t2 <= 1:
            incidence_angle = (np.pi / 2) - np.arccos(np.dot(normalize_vector(self.direction), normalize_vector(v2)))
            new_direction = self.origin + t1 * self.direction
            return [new_direction, incidence_angle, (normal1, normal2)]
        else:
            return [None, None, None]


    def intersect_triangle(self, triangle_vertices):
        edges = [(triangle_vertices[0], triangle_vertices[1]), (triangle_vertices[1], triangle_vertices[2]), (triangle_vertices[2], triangle_vertices[0])]
        intersections = [self.ray_intersect_segment(vertex1, vertex2) for (vertex1,vertex2) in edges]
        # print(intersections)
        distance_to_intersections = []

        for intersection, _, _ in intersections:
            if intersection is None:
                distance_to_intersections.append(np.Inf)
            else:
                distance_to_intersections.append(np.sqrt(np.sum(np.square(intersection - self.origin))))

        distance_to_intersections = np.array(distance_to_intersections)
        print(distance_to_intersections, intersections[np.argmin(distance_to_intersections)], edges[np.argmin(distance_to_intersections)])
        if np.min(distance_to_intersections) == np.Inf:
            return [None, None, None]
        else:
            return intersections[np.argmin(distance_to_intersections)]
        
    def refract(self, intersection, incidence_angle, new_medium, normals):
        theta1 = incidence_angle
        n1 = self.properties.refractive_index[self.medium]
        n2 = self.properties.refractive_index[new_medium]
        
        sin_theta2 = n1 * np.sin(theta1) / n2

        if np.abs(sin_theta2) > 1:
            return self.reflect(intersection, normals, new_medium)

        theta2 = np.arcsin(sin_theta2)

        rotation_matrix = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
        new_direction = normalize_vector(np.dot(rotation_matrix, self.direction))

        return Ray(intersection, new_direction, self.properties, new_medium)
    
    def reflect(self, intersection, normals, new_medium):
        # Pick n pointing away from new medium
        if np.dot(self.direction, normals[0]) < 0:
            n = normals[0] 
        else:
            n = normals[1]
        
        # r = i-2*(v . n)*n
        new_direction = normalize_vector(self.direction - 2 * np.dot(self.direction,n) * n)

        print(f"Reflecting: {self.direction} at {intersection} to {new_direction}.")

        return Ray(intersection, new_direction, self.properties, new_medium)

def normalize_vector(v):
    return v/np.linalg.norm(v)

def vector_magnitude(v):
    return hypot(v[0], v[1])

triangle_vertices = np.array([[0, 0], [200, 0], [100, 200]])
light_source = np.array([0, 100])

ray_direction = np.array([1, 0])

red_ray = Ray(light_source, ray_direction, red_light, 'air')
orange_ray = Ray(light_source, ray_direction, orange_light, 'air')
yellow_ray = Ray(light_source, ray_direction, yellow_light, 'air')
green_ray = Ray(light_source, ray_direction, green_light, 'air')
blue_ray = Ray(light_source, ray_direction, blue_light, 'air')
indigo_ray = Ray(light_source, ray_direction, indigo_light, 'air')
violet_ray = Ray(light_source, ray_direction, violet_light, 'air')

rays = [red_ray, orange_ray, yellow_ray, green_ray, blue_ray, indigo_ray, violet_ray]

for i, ray in enumerate(rays):
    # ray_origin = [light_source[0] + i * 0.1, light_source[1]]
    current_ray = ray
    for k in range(MAX_BOUNCE):
        if k == 0:
            new_medium = 'glass'
        elif k >= 1:
            new_medium = 'air'

        intersection, incidence_angle, normals = current_ray.intersect_triangle(triangle_vertices)
        print(f"k={k}, intersect={intersection}, direction={current_ray.direction}")
        plt.plot(current_ray.origin[0], current_ray.origin[1], marker='o', color=current_ray.properties.color)

        if intersection is not None:
            plt.plot([current_ray.origin[0],intersection[0]],[current_ray.origin[1],intersection[1]], color=current_ray.properties.color)

            current_ray = current_ray.refract(intersection, incidence_angle, new_medium, normals)
        else:
            # If ray doesn't refract, end loop
            plt.plot([current_ray.origin[0],current_ray.origin[0] + current_ray.direction[0]*100],[current_ray.origin[1],current_ray.origin[1]+ current_ray.direction[1]*100], color=current_ray.properties.color)
            break
            # plt.plot(intersection[0], intersection[1], marker='x', color='red')

plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color='black')
plt.plot([triangle_vertices[0, 0], triangle_vertices[-1, 0]], [triangle_vertices[0, 1], triangle_vertices[-1, 1]], color='black')
plt.axis('equal')
plt.show()

