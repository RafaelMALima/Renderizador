#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Rafael Lima
Disciplina: Computação Gráfica
Data: 09/08/24 dd/mm/yy
"""

import time

from numpy._typing import NDArray         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy


## agradecimentos especiais a pedro barao por me forneces essa merda
def t_area(p0,p1,p2):
    x1,y1 = p0
    x2,y2 = p1
    x3,y3 = p2
    return 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))

def t_bar_coords(p0, p1, p2, p):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        x, y = p
        a_total = t_area([x0, y0], [x1, y1], [x2, y2])
        a0 = t_area([x1, y1], [x2, y2], [x, y])
        a1 = t_area([x2, y2], [x0, y0], [x, y])
        a2 = t_area([x0, y0], [x1, y1], [x, y])
        alpha = abs(a0 / a_total)
        beta = abs(a1 / a_total)
        gamma = abs(a2 / a_total)
        return alpha,beta,gamma

#GPT pq preguiça
def calculate_triangle_normal(v0, v1, v2):
    """
    Calculate the normal vector of a triangle defined by vertices v0, v1, and v2.

    Parameters:
    v0, v1, v2 : array-like
        The coordinates of the triangle's vertices. Each can be a list, tuple, or NumPy array with three elements.

    Returns:
    normal : ndarray
        A unit normal vector perpendicular to the plane of the triangle.
    """
    # Convert vertices to NumPy arrays
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the cross product of the edge vectors
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm
    else:
        # Handle degenerate triangle (zero area)
        normal = np.array([0.0, 0.0, 0.0])

    if normal is list[list]:
        return normal[0]
    else:
        return normal

def calculate_color(material_emissive_color,
                    material_diffuse_color,
                    material_specular_color,
                    material_ambientIntensity,
                    light_color,
                    light_intensity,
                    light_ambient_intensity,
                    light_direction,
                    normal_direction,
                    point_to_viewer_normalized,
                    shininess):

    light_direction = np.array([-a for a in light_direction])
    ambient = light_ambient_intensity*np.array(material_diffuse_color)*material_ambientIntensity
    diffuse = light_intensity*np.array(material_diffuse_color)*(np.dot(normal_direction,light_direction))
    specular = light_intensity*np.array(material_specular_color)*((np.dot(normal_direction,((light_direction+point_to_viewer_normalized)/abs(light_direction+point_to_viewer_normalized))))**(shininess*128))
    specular = (0.0, 0.0, 0.0)
    #print(material_emissive_color, material_diffuse_color, material_specular_color, material_ambientIntensity)
    #print(ambient, diffuse, specular)
    I_rgb = np.array(material_emissive_color) + light_color*(ambient + diffuse + specular)
    #print(I_rgb)
    return np.clip(I_rgb*256,0,255)

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    start_time = 0

    camera_transform_matrix = []
    transform_stack : list[NDArray] = []
    normals_transform_stack : list[NDArray] = []

    screen_matrix = []
    z_buffer = [[0]*width]*height
    supersampling_buffer = np.zeros((1,1,3), dtype=np.uint8)
    z_buffer = np.array(np.inf)

    texture = None
    texture_mipmap = []

    light: dict = {
             "ambientLight":None,
             "directionalLight":None,
             "light_color": None,
             "light_intensity":None,
             "light_ambient_intensity":None,
             "light_point":None,
             "light_direction":None
         }



    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definir parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.screen_matrix = np.array([[GL.width/2,0,0,GL.width/2],
                     [0,-GL.height/2,0,GL.height/2],
                     [0,0,1,0],
                     [0,0,0,1]])

        GL.supersampling_buffer = np.zeros((GL.width*2, GL.height*2, 3), dtype=np.uint8)
        GL.z_buffer = np.inf * np.ones((GL.width*2, GL.height*2))
        GL.light =  {
            "ambientLight":None,
            "directionalLight":None,
            "light_color": None,
            "light_intensity":None,
            "light_ambient_intensity":None,
            "light_point":None
        }


    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        for i in range(0,len(point),2):
            pos_x = point[i]
            pos_y = point[i+1]
            cor = colors['emissiveColor']
            gpu.GPU.draw_pixel([int(pos_x), int(pos_y)], gpu.GPU.RGB8, [cor[0]*255,cor[1]*255,cor[2]*255])  # altera pixel (u, v, tipo, r, g, b)

        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        cor = colors['emissiveColor']

        for i in range(0,len(lineSegments) - 2,2):
            pos_x0 = lineSegments[i]
            pos_y0 = lineSegments[i+1]
            pos_x1 = lineSegments[i+2]
            pos_y1 = lineSegments[i+3]
            if pos_x0 == pos_x1:
                s = np.inf
            else:
                s = (pos_y1 - pos_y0)/(pos_x1 - pos_x0)
            if s < 1 and s > -1:
                if pos_x0 > pos_x1:
                    pos_x0, pos_x1 = pos_x1, pos_x0
                    pos_y0, pos_y1 = pos_y1, pos_y0
                u = pos_x0
                v = pos_y0
                while u < pos_x1:
                    gpu.GPU.draw_pixel([int(u), int(v)], gpu.GPU.RGB8, [cor[0]*255,cor[1]*255,cor[2]*255])  # altera pixel (u, v, tipo, r, g, b)
                    u += 1
                    v += s
            else:
                s = 1/s
                if pos_y0 > pos_y1:
                    pos_x0, pos_x1 = pos_x1, pos_x0
                    pos_y0, pos_y1 = pos_y1, pos_y0
                u = pos_x0
                v = pos_y0
                while v < pos_y1:
                    gpu.GPU.draw_pixel([int(u), int(v)], gpu.GPU.RGB8, [cor[0]*255,cor[1]*255,cor[2]*255])  # altera pixel (u, v, tipo, r, g, b)
                    v += 1
                    u += s

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        pos_x = GL.width//2
        pos_y = GL.height//2
        center = [pos_x, pos_y]
        bounding_box = [int(center[0]-radius), int(center[1]-radius), int(center[0] + radius), int(center[1] + radius)]
        for i in range(len(bounding_box)):
            if bounding_box[i] <= 0:
                bounding_box[i] = 0

        for j in range(bounding_box[1], bounding_box[3]):
            for i in range(bounding_box[0], bounding_box[2]):
                p_x = bounding_box[0] + i
                p_y = bounding_box[1] + j
                gpu.GPU.draw_pixel([p_x,p_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)


    @staticmethod
    def triangleSet2D(vertices, colors, barycenter = None, z_vals = None, colorPerVertex = False, vertexColors = None,
                      textCoords = None, texture = None, normals = None):
        """Função usada para renderizar TriangleSet2D."""


        #print(z_vals)
        if len(vertices) < 5:
            print("ERROR NO TRIANGLES SENT")
            return

        #print(textCoords)
        has_texture = False
        if textCoords is not None and texture is not None:
            has_texture = True
            # Extract texture coordinates for each vertex
            t1 = textCoords[0]
            t2 = textCoords[1]
            t3 = textCoords[2]
            texture_image = texture

        color = np.array(colors["emissiveColor"]) * 255

        z_index = 0
        for i in range(0, len(vertices), 6):
            tri = vertices[i: i + 6]

            z_val = None
            if z_vals is not None:
                z_val = z_vals[i//2:i//2+3]

            if len(tri) != 6:
                return

            # Extrair as coordenadas dos vértices do triângulo no espaço da tela
            p1 = [tri[0], tri[1]]
            p2 = [tri[2], tri[3]]
            p3 = [tri[4], tri[5]]

            xs = [p1[0], p2[0], p3[0]]
            ys = [p1[1], p2[1], p3[1]]

            # Calcular a caixa delimitadora (bounding box) do triângulo para a 
            min_x = int(min(xs))
            max_x = int(max(xs))
            min_y = int(min(ys))
            max_y = int(max(ys))

            #screenspace
            bounding_box = [min_x, max_x, min_y, max_y]


            #supersampling space
            s_p1 = [tri[0]*2, tri[1]*2]
            s_p2 = [tri[2]*2, tri[3]*2]
            s_p3 = [tri[4]*2, tri[5]*2]

            s_bounding_box = [2*c for c in bounding_box]

            z1, z2, z3 = [1,1,1]
            if z_vals is not None:
                z1, z2, z3 = z_vals[z_index:z_index+3]

            # Iterar sobre cada pixel dentro da bounding box
            for x in range(s_bounding_box[0], s_bounding_box[1] + 1):
                for y in range(s_bounding_box[2], s_bounding_box[3] + 1):
                    # Coordenadas do ponto central do pixel
                    s_px = x + 0.5
                    s_py = y + 0.5

                    # Cálculo das funções de linha para cada aresta do triângulo
                    L1 = (s_p2[1] - s_p1[1]) * s_px - (s_p2[0] - s_p1[0]) * s_py + s_p1[1] * (s_p2[0] - s_p1[0]) - s_p1[0] * (s_p2[1] - s_p1[1])
                    L2 = (s_p3[1] - s_p2[1]) * s_px - (s_p3[0] - s_p2[0]) * s_py + s_p2[1] * (s_p3[0] - s_p2[0]) - s_p2[0] * (s_p3[1] - s_p2[1])
                    L3 = (s_p1[1] - s_p3[1]) * s_px - (s_p1[0] - s_p3[0]) * s_py + s_p3[1] * (s_p1[0] - s_p3[0]) - s_p3[0] * (s_p1[1] - s_p3[1])

                    # Verificar se o ponto está dentro do triângulo
                    if (L1 >= 0 and L2 >= 0 and L3 >= 0) or (L1 <= 0 and L2 <= 0 and L3 <= 0):
                        if 0 <= x < GL.width*2 and 0 <= y < GL.height*2:
                            alpha, beta, gamma = t_bar_coords(s_p1, s_p2, s_p3, [s_px, s_py])
                            if alpha < 0 or beta < 0 or gamma < 0:
                                continue

                            z = 1/(alpha * z1 + beta * z2 + gamma * z3)

                            if z < GL.z_buffer[x][y]:
                                GL.z_buffer[x][y] = z
                            else:
                                #skip subpixel if not ahead of z buffer
                                continue

                            # Avoid division by zero
                            inv_z1 = 1.0 / z1 if z1 != 0 else 0.0
                            inv_z2 = 1.0 / z2 if z2 != 0 else 0.0
                            inv_z3 = 1.0 / z3 if z3 != 0 else 0.0

                            if vertexColors is not None and z_vals is not None:
                                inv_z = alpha * inv_z1 + beta * inv_z2 + gamma * inv_z3
                                if inv_z == 0:
                                    continue  # Skip this pixel to avoid division by zero

                                # Interpolate color components with perspective correction
                                r = (alpha * vertexColors[0][0] * inv_z1 + beta * vertexColors[1][0] * inv_z2 + gamma * vertexColors[2][0] * inv_z3) / inv_z
                                g = (alpha * vertexColors[0][1] * inv_z1 + beta * vertexColors[1][1] * inv_z2 + gamma * vertexColors[2][1] * inv_z3) / inv_z
                                b = (alpha * vertexColors[0][2] * inv_z1 + beta * vertexColors[1][2] * inv_z2 + gamma * vertexColors[2][2] * inv_z3) / inv_z

                                pixel_color = np.array([r, g, b])
                                pixel_color = np.clip(pixel_color, 0, 255).astype(np.uint8)
                                GL.supersampling_buffer[x][y] = pixel_color
                            else:
                                if normals is not None:
                                    #Calculate light direction
                                    """
                                    GL.Light = {
                                     "ambientLight":None,
                                     "directionalLight":None,
                                     "light_color": None,
                                     "light_intensity":None,
                                     "light_ambient_intensity":None,
                                     "light_point":None,
                                     "light_direction":None
                                    }
                                    def calculate_color(material_emissive_color,
                                                        material_diffuse_color,
                                                        material_specular_color,
                                                        material_ambientIntensity,
                                                        light_color,
                                                        light_intensity,
                                                        light_ambient_intensity,
                                                        light_direction,
                                                        normal_direction,
                                                        point_to_viewer_normalized,
                                                        shininess):
                                    """
                                    point_to_viewer = barycenter  - GL.position
                                    point_to_viewer_normalized= np.linalg.norm(point_to_viewer)
                                    if point_to_viewer_normalized!= 0:
                                        point_to_viewer = point_to_viewer / point_to_viewer_normalized
                                    else:
                                        # Handle degenerate triangle (zero area)
                                        normal = np.array([0.0, 0.0, 0.0])
                                    if GL.light["ambientLight"]:
                                        light_direction = GL.light["light_direction"]
                                        color = calculate_color(colors["emissiveColor"],
                                                        colors["diffuseColor"],
                                                        colors["specularColor"],
                                                        GL.light["light_ambient_intensity"],                                                        GL.light["light_color"],
                                                        GL.light["light_intensity"],
                                                        GL.light["light_ambient_intensity"],
                                                        light_direction,
                                                        normals,
                                                        point_to_viewer_normalized,
                                                        colors["shininess"]
                                                        )
                                        GL.supersampling_buffer[x][y] = color
                                    elif GL.light["directionalLight"]:
                                        point_to_viewer_normalized = barycenter  - GL.position
                                        light_direction = GL.light["light_point"] - barycenter
                                        GL.supersampling_buffer[x][y] = [int(a*255) for a in np.array(colors["diffuseColor"])]
                                        color = calculate_color(colors["emissiveColor"],
                                                        colors["diffuseColor"],
                                                        colors["specularColor"],
                                                        GL.light["light_ambient_intensity"],
                                                        GL.light["light_color"],
                                                        GL.light["light_intensity"],
                                                        GL.light["light_ambient_intensity"],
                                                        light_direction,
                                                        normals,
                                                        point_to_viewer_normalized,
                                                        colors["shininess"]
                                                        )
                                        GL.supersampling_buffer[x][y] = color
                                    normal_color = [int((c + 1)*0.5*255) for c in normals]
                                    #GL.supersampling_buffer[x][y] = normal_color
                                    GL.supersampling_buffer[x][y] = color
                                else:
                                    GL.supersampling_buffer[x][y] = color


            for x in range(bounding_box[0], bounding_box[1] + 1):
                for y in range(bounding_box[2], bounding_box[3] + 1):
                    sub_pixels = GL.supersampling_buffer[2*x:2*x+2, 2*y:2*y+2]
                    #if (x,y) == (44,47):
                        #print(f"sub_pixels{sub_pixels}")
                    super_colors = sub_pixels.mean(axis=(0, 1)).astype(np.uint8)
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, super_colors)

            z_index += 3

    @staticmethod
    def triangleSet(point, colors, colorPerVertex=False, vertexColors=None, texCoord=None, texture=None):
        """Function to render TriangleSet."""

        for i in range(len(point) // 9):
            p = point[i*9:i*9+9]
            p_a = [p[0], p[1], p[2]]
            p_b = [p[3], p[4], p[5]]
            p_c = [p[6], p[7], p[8]]

            triangle_matrix_no_transform = np.array([
                [p_a[0], p_b[0], p_c[0]],
                [p_a[1], p_b[1], p_c[1]],
                [p_a[2], p_b[2], p_c[2]],
                [1., 1., 1.]])


            # Apply all model transformations
            transform_matrix = np.identity(4)
            for matrix in reversed(GL.transform_stack):
                transform_matrix = matrix @ transform_matrix

            """
            normal_transforms = np.identity(4)
            for matrix in reversed(GL.normals_transform_stack):
                normals_transforms = normals @ matrix

            # normal in world space
            n_to_world = multiply_mats(GL.normal_transform_stack)
            n = ns[i]
            n.append(1.0)
            world_n = np.array(n)
            world_n = n_to_world @ world_n
            """


            # Transform to world space
            triangle_matrix_worldspace = transform_matrix @ triangle_matrix_no_transform

            #worldspace vertices after transform
            v0 = [triangle_matrix_worldspace[0,0], triangle_matrix_worldspace[1,0], triangle_matrix_worldspace[2,0]]
            v1 = [triangle_matrix_worldspace[0,1], triangle_matrix_worldspace[1,1], triangle_matrix_worldspace[2,1]]
            v2 = [triangle_matrix_worldspace[0,2], triangle_matrix_worldspace[1,2], triangle_matrix_worldspace[2,2]]
            #barycenter for point in worldspace to cameraspace
            barycenter = (np.array(v0)+np.array(v1)+np.array(v2)) / 3

            #normals
            normals = calculate_triangle_normal(v0, v1, v2)

            # Apply view (camera) transformation to get view space coordinates
            look_at_p = GL.view_matrix @ triangle_matrix_worldspace

            # Extract z-values from the view space coordinates
            z_vals = look_at_p[2, :].tolist()[0]  # Convert to list

            # Apply projection matrix
            triangle_matrix_unnormalized = GL.perspective_matrix @ look_at_p

            # Perspective divide
            triangle_matrix = triangle_matrix_unnormalized / triangle_matrix_unnormalized[3, :]

            # Apply screen transformation
            triangle_matrix_cameraspace = GL.screen_matrix @ triangle_matrix

            triangle_array = np.asarray(triangle_matrix_cameraspace)

            triangle_points = [
                triangle_array[0][0],
                triangle_array[1][0],
                triangle_array[0][1],
                triangle_array[1][1],
                triangle_array[0][2],
                triangle_array[1][2]]

            tri_vertex_colors = None
            if colorPerVertex and vertexColors:
                tri_vertex_colors = vertexColors[i*3:i*3+3]

            texCoords=None
            if texCoord is not None:
                t_a = texCoord[i * 6:i * 6 + 2]
                t_b = texCoord[i * 6 + 2:i * 6 + 4]
                t_c = texCoord[i * 6 + 4:i * 6 + 6]
                texCoords = [t_a, t_b, t_c]
            # Pass the z-values to triangleSet2D

            

            GL.triangleSet2D(triangle_points, colors,barycenter=barycenter, z_vals=z_vals, colorPerVertex=colorPerVertex, vertexColors=tri_vertex_colors, textCoords=texCoords, texture=texture, normals=normals)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Function to handle Viewpoint node."""
        GL.position = position
        GL.orientation = orientation
        GL.fov = fieldOfView

        camera_translation = np.matrix([
            [1., 0., 0., position[0]],
            [0., 1., 0., position[1]],
            [0., 0., 1., position[2]],
            [0., 0., 0., 1.],
        ])

        rotation_matrix = GL.calculate_rotation_matrix(orientation[:3], orientation[3])

        # Invert the camera transformation to get the view matrix
        view_matrix = np.linalg.inv(camera_translation @ rotation_matrix)
        GL.view_matrix = view_matrix  # Store the view matrix

        aspect_ratio = GL.width / GL.height
        near = GL.near
        far = GL.far
        top = near * np.tan(fieldOfView / 2)
        right = top * aspect_ratio

        perspective_matrix = np.matrix([
            [near / right, 0.0, 0.0, 0.0],
            [0.0, near / top, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * (far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ])

        GL.perspective_matrix = perspective_matrix

        # Update camera_transform_matrix if needed
        GL.camera_transform_matrix = perspective_matrix @ view_matrix

    @staticmethod
    def calculate_rotation_matrix(rot_v, theta):
        cos = np.cos(theta)
        sin = np.sin(theta)

        x, y, z = rot_v

        rotation_matrix = np.array([
            [cos + x**2 * (1 - cos),       x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin, 0],
            [y * x * (1 - cos) + z * sin,  cos + y**2 * (1 - cos),      y * z * (1 - cos) - x * sin, 0],
            [z * x * (1 - cos) - y * sin,  z * y * (1 - cos) + x * sin, cos + z**2 * (1 - cos),      0],
            [0,                            0,                           0,                            1]
        ])
        return rotation_matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        position = translation
        translation_matrix = np.matrix([
            [1.,0.,0.,position[0]],
            [0.,1.,0.,position[1]],
            [0.,0.,1.,position[2]],
            [0.,0.,0.,1.],
            ])

        scale_matrix = np.matrix([
            [scale[0],0.,0.,0.],
            [0.,scale[1],0.,0.],
            [0.,0.,scale[2],0.],
            [0.,0.,0.,1.],
            ])

        if rotation is not None:
            rotation_matrix = GL.calculate_rotation_matrix(rotation[:3],rotation[3])
        else:
            rotation_matrix = np.identity(4)

        # Correct the order of transformations
        transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix

        GL.transform_stack.append(transform_matrix)
        GL.normals_transform_stack.append(rotation_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        GL.transform_stack.pop()
        GL.normals_transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        vertices = []                      
        for i in range(0,len(point)-3,3):
            v1 = point[i:i+3]
            v2 = point[i+3:i+6]
            v3 = point[i+6:i+9] if i+6 < len(point) else point[0:3]
            coords = v1 + v2 + v3
            vertices.extend(coords)

        GL.triangleSet(vertices, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors, color_index=None):
        """Função usada para renderizar IndexedTriangleStripSet."""
        strips = []
        current_strip = []
        for idx in index:
            if idx == -1:
                if current_strip:
                    strips.append(current_strip)
                    current_strip = []
            else:
                current_strip.append(idx)
        if current_strip:
            strips.append(current_strip)
        
        # Para cada tira, gerar os triângulos
        for strip in strips:
            if len(strip) < 3:
                continue  # É necessário pelo menos 3 vértices para formar um triângulo
            triangles = []
            for i in range(len(strip) - 2):
                a = strip[i]
                b = strip[i+1]
                c = strip[i+2]
                # Extrair as coordenadas dos pontos
                coords = point[3*a : 3*a+3] + point[3*b : 3*b+3] + point[3*c : 3*c+3]
                triangles.extend(coords)
            # Chamar a função para desenhar os triângulos
            GL.triangleSet(triangles, colors)       

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        scale_x, scale_y, scale_z = size
        vertices = [
            # Front face
            -scale_x / 2, -scale_y / 2, scale_z / 2,
            scale_x / 2, -scale_y / 2, scale_z / 2,
            scale_x / 2, scale_y / 2, scale_z / 2,
            -scale_x / 2, scale_y / 2, scale_z / 2,
            # Back face
            -scale_x / 2, -scale_y / 2, -scale_z / 2,
            scale_x / 2, -scale_y / 2, -scale_z / 2,
            scale_x / 2, scale_y / 2, -scale_z / 2,
            -scale_x / 2, scale_y / 2, -scale_z / 2,
        ]

        # Define the 12 triangles of the box
        triangles = [
            # Front face
            0, 1, 2,
            0, 2, 3,
            # Back face
            4, 6, 5,
            4, 7, 6,
            # Top face
            3, 2, 6,
            3, 6, 7,
            # Bottom face
            0, 4, 1,
            1, 4, 5,
            # Right face
            1, 5, 2,
            2, 5, 6,
            # Left face
            0, 3, 7,
            0, 7, 4,
        ]
        GL.indexedTriangleStripSet(vertices, triangles, colors)



    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Function used to render IndexedFaceSet with texture mapping."""

        # Load the texture image if not already loaded
        if current_texture:
            GL.texture = gpu.GPU.load_texture(current_texture[0])
            GL.texture_mipmap = generate_mipmap(GL.texture)
        else:
            GL.texture = None

        # Process coordIndex into faces
        faces = []
        current_face = []
        for idx in coordIndex:
            if idx == -1:
                if current_face:
                    faces.append(current_face)
                current_face = []
            else:
                current_face.append(idx)
        if current_face:
            faces.append(current_face)

        # Process texCoordIndex into texture coordinate indices
        if texCoordIndex:
            tex_faces = []
            current_tex_face = []
            for idx in texCoordIndex:
                if idx == -1:
                    if current_tex_face:
                        tex_faces.append(current_tex_face)
                    current_tex_face = []
                else:
                    current_tex_face.append(idx)
            if current_tex_face:
                tex_faces.append(current_tex_face)
        else:
            tex_faces = None

        # Process each face
        for face_idx, face in enumerate(faces):
            if len(face) < 3:
                continue  # Need at least 3 vertices to form a polygon

            # Get the corresponding texture indices for this face
            if tex_faces:
                tex_face = tex_faces[face_idx]
            else:
                tex_face = None

            # Triangulate the face using the fan method
            v0 = face[0]
            for i in range(1, len(face) - 1):
                v1 = face[i]
                v2 = face[i + 1]

                # Get the coordinates of the vertices
                coords = []
                for vi in [v0, v1, v2]:
                    x = coord[3 * vi]
                    y = coord[3 * vi + 1]
                    z = coord[3 * vi + 2]
                    coords.extend([x, y, z])

                # Get the texture coordinates for the vertices
                if texCoord and tex_face:
                    t0 = texCoord[2 * tex_face[0]:2 * tex_face[0] + 2]
                    t1 = texCoord[2 * tex_face[i]:2 * tex_face[i] + 2]
                    t2 = texCoord[2 * tex_face[i + 1]:2 * tex_face[i + 1] + 2]
                    texCoords = [t0, t1, t2]
                else:
                    texCoords = None

                # Handle per-vertex colors
                if colorPerVertex:
                    # Similar handling for vertex colors as before
                    pass  # Implement color handling if needed

                GL.triangleSet(coords, colors, texCoord=texCoords, texture=GL.texture)

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores
        segments = 20  # Define um valor fixo de segmentos para a esfera
        vertices = []
        triangles = []

        for i in range(segments + 1):
            theta = i * math.pi / segments  # de 0 a pi
            for j in range(segments + 1):
                phi = j * 2 * math.pi / segments  # de 0 a 2pi
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                vertices.extend([x, y, z])

                if i < segments and j < segments:
                    first = i * (segments + 1) + j
                    second = first + segments + 1
                    triangles.extend([first, second, first + 1, second, second + 1, first + 1])

        GL.indexedTriangleStripSet(vertices, triangles, colors)

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        print("Cone: bottomRadius = {0}, height = {1}".format(bottomRadius, height))  # imprime no terminal o raio e altura
        print("Cone: colors = {0}".format(colors))  # imprime no terminal as cores

        segments = 20  # Define um valor fixo de segmentos para o cone
        vertices = [0, height / 2, 0]  # vértice do topo do cone
        triangles = []

        for i in range(segments):
            theta = i * 2 * math.pi / segments
            x = bottomRadius * math.cos(theta)
            z = bottomRadius * math.sin(theta)
            vertices.extend([x, -height / 2, z])

        for i in range(1, segments):
            triangles.extend([0, i, i + 1])
        triangles.extend([0, segments, 1])  # fecha a base

        GL.indexedTriangleStripSet(vertices, triangles, colors)

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        print("Cylinder: radius = {0}, height = {1}".format(radius, height))  # imprime no terminal o raio e altura
        print("Cylinder: colors = {0}".format(colors))  # imprime no terminal as cores

        segments = 12  # Define um valor fixo de segmentos para o cilindro
        vertices = []
        triangles = []

        # Cria os vértices para os dois círculos da base
        for i in range(segments):
            theta = i * 2 * math.pi / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.extend([x, -height / 2, z])  # círculo inferior
            vertices.extend([x, height / 2, z])  # círculo superior

        # Cria os triângulos das faces laterais
        for i in range(segments):
            lower1 = 2 * i
            upper1 = lower1 + 1
            lower2 = 2 * ((i + 1) % segments)
            upper2 = lower2 + 1
            triangles.extend([lower1, upper1, lower2, upper1, upper2, lower2])

        # Cria as tampas inferior e superior
        for i in range(1, segments - 1):
            # Inferior
            triangles.extend([0, 2 * i, 2 * (i + 1)])
            # Superior
            triangles.extend([1, 2 * (i + 1) + 1, 2 * i + 1])

        GL.indexedTriangleStripSet(vertices, triangles, colors)
    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal
        GL.light["ambientLight"] = True
        GL.light["directionalLight"] = False
        GL.light["light_color"] = color
        GL.light["light_intensity"] = intensity
        GL.light["light_ambient_intensity"] = ambientIntensity
        GL.light["light_direction"] = direction


    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal
        GL.light["ambientLight"] = False
        GL.light["directionalLight"] = True
        GL.light["light_color"] = color
        GL.light["light_intensity"] = intensity
        GL.light["light_ambient_intensity"] = ambientIntensity
        GL.light["light_point"] = location

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):

        GL.supersampling_buffer = np.zeros((GL.width*2, GL.height*2, 3), dtype=np.uint8)
        GL.z_buffer = np.full((GL.width * 2, GL.height * 2), -np.inf)

        # Esse método já está implementado para os alunos como exemplo
        epoch = (time.time())  # time in seconds since the epoch as a floating point number.
        if loop:
            relative_time = ((epoch - GL.start_time) % cycleInterval) / cycleInterval
        else:
            relative_time = np.clip(epoch - GL.start_time / cycleInterval, 0, 1)

    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        key = np.array(key)
        keyValue = np.array(keyValue)

        k_i_before = 0
        for k_i_before in range(len(key) - 1): # Find the key interval
            if key[k_i_before] <= set_fraction <= key[k_i_before + 1]:
                k_i_before = k_i_before
                k_i_plus = k_i_before + 1
                break

        key_value_parsed = keyValue.reshape(-1, 3)
        #print(f"SplinePositionInterpolator : key_value_parsed = \n{key_value_parsed}")

        if set_fraction == key[k_i_before]:
            return key_value_parsed[k_i_before:k_i_before+1]
        if set_fraction == key[k_i_plus]:
            return key_value_parsed[k_i_plus:k_i_plus+1]
        
        delta_key = key[k_i_plus] - key[k_i_before]
        s = (set_fraction - key[k_i_before])/delta_key
        s_m = np.array([
            s**3,
            s**2,
            s,
            1
        ])

        # Handle boundary cases for deriv_0
        if k_i_before == 0:
            if closed:
                deriv_0 = (key_value_parsed[-1] - key_value_parsed[1]) * 0.5
            else:
                deriv_0 = np.array([0, 0, 0])
        else:
            deriv_0 = (key_value_parsed[k_i_before-1] - key_value_parsed[k_i_before+1]) * 0.5


        # Handle boundary cases for delta_value_1
        if k_i_plus == len(key) - 1:
            if closed:
                deriv_1 = (key_value_parsed[0] - key_value_parsed[-2]) * 0.5
            else:
                deriv_1 = np.array([0, 0, 0])
        else:
            deriv_1 = (key_value_parsed[k_i_plus-1] - key_value_parsed[k_i_plus+1]) * 0.5

        

        c = np.array([
            key_value_parsed[k_i_before],
            key_value_parsed[k_i_plus],
            deriv_0,
            deriv_1
        ])

        # print(f"SplinePositionInterpolator : k_i_before = {k_i_before}")
        # print(f"SplinePositionInterpolator : k_i_plus = {k_i_plus}")
        # print(f"SplinePositionInterpolator : set_fraction = {set_fraction}")
        # print(f"SplinePositionInterpolator : key = {key}")
        # print(f"SplinePositionInterpolator : keyValue = {keyValue}")
        # print(f"SplinePositionInterpolator : closed = {closed}")
        # print(f"SplinePositionInterpolator : c = \n{c}")

        # Calcular a interpolação
        value_changed = s_m @ np.array([
        [ 2, -2,  1,  1],
        [-3,  3, -2, -1],
        [ 0,  0,  1,  0],
        [ 1,  0,  0,  0]
    ]) @ c

        #print(f"SplinePositionInterpolator : value_changed = \n{value_changed}")

        return value_changed
    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """ print(f"OrientationInterpolator : key_value_parsed = \n{key_value_parsed}")
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))
        print("OrientationInterpolator : keyValue = {0}".format(keyValue)) """

    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
