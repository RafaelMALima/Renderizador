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

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    camera_transform_matrix = []
    transform_stack : list[NDArray] = []

    screen_matrix = []
    z_buffer = [[0]*width]*height
    supersampling_buffer = np.zeros((1,1,3), dtype=np.uint8)
    z_buffer = np.array(np.inf)



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
        GL.z_buffer = - np.inf * np.ones((GL.width*2, GL.height*2))


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
    def triangleSet2D(vertices, colors, z_vals = None, colorPerVertex = False, vertexColors = None):
        """Função usada para renderizar TriangleSet2D."""

        if len(vertices) < 5:
            print("ERROR NO TRIANGLES SENT")
            return


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

            z1, z2, z3 = [0,0,0]
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

                            """
                            if z > GL.z_buffer[x][y]:
                                one_over_z = alpha * one_over_z1 + beta * one_over_z2 + gamma * one_over_z3
                                if one_over_z == 0:
                                    continue  # Avoid division by zero

                                r = vertexColors[3*i][0] * alpha + vertexColors[3*i+1][0] * beta + vertexColors[3*i+2][0] * gamma
                                g = vertexColors[3*i][1] * alpha + vertexColors[3*i+1][1] * beta + vertexColors[3*i+2][1] * gamma
                                b = vertexColors[3*i][2] * alpha + vertexColors[3*i+1][2] * beta + vertexColors[3*i+2][2] * gamma
                                cr = z * r/z_val[0]
                                cg = z * g/z_val[1]
                                cb = z * b/z_val[2]
                                color = [int(cr), int(cg), int(cb)]
                            """
                            if z > GL.z_buffer[x][y]:
                                GL.z_buffer[x][y] = z
                            else:
                                #skip subpixel if not ahead of z buffer
                                continue

                            GL.supersampling_buffer[x][y] = color

            for x in range(bounding_box[0], bounding_box[1] + 1):
                for y in range(bounding_box[2], bounding_box[3] + 1):
                    sub_pixels = GL.supersampling_buffer[2*x:2*x+2, 2*y:2*y+2]
                    super_colors = sub_pixels.mean(axis=(0, 1)).astype(np.uint8)
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, super_colors)

            z_index += 3

    @staticmethod
    def triangleSet(point, colors, colorPerVertex = False, vertexColors = None):
        """Função usada para renderizar TriangleSet."""

        for i in range(len(point) // 9):
            p = point[i*9:i*9+9]
            p_a : list[float] = [p[0], p[1], p[2]]
            p_b : list[float] = [p[3], p[4], p[5]]
            p_c : list[float] = [p[6], p[7], p[8]] 

            triangle_matrix_no_transform = np.array([
                    [p_a[0], p_b[0], p_c[0]],
                    [p_a[1], p_b[1], p_c[1]],
                    [p_a[2], p_b[2], p_c[2]],
                    [1.,1.,1.]])

            #Apply all transforms
            transform_matrix = np.identity(4)
            for matrix in reversed(GL.transform_stack):
                transform_matrix = matrix @ transform_matrix

            triangle_matrix_worldspace = transform_matrix @ triangle_matrix_no_transform

            camera_transform_matrix = GL.camera_transform_matrix

            triangle_matrix_unnormalized = camera_transform_matrix @ triangle_matrix_worldspace

            # Perspective divide
            triangle_matrix = triangle_matrix_unnormalized / triangle_matrix_unnormalized[3,:]

            screen_matrix = GL.screen_matrix

            triangle_matrix_cameraspace = screen_matrix @ triangle_matrix

            triangle_array = np.asarray(triangle_matrix_cameraspace)

            triangle_points = [triangle_array[0][0],
                                triangle_array[1][0],
                                triangle_array[0][1],
                                triangle_array[1][1],
                                triangle_array[0][2],
                                triangle_array[1][2]]

            GL.triangleSet2D(triangle_points, colors, colorPerVertex=colorPerVertex, vertexColors=vertexColors)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        GL.position = position
        GL.orientation = orientation
        GL.fov = fieldOfView

        camera_translation = np.matrix([
            [1.,0.,0.,position[0]],
            [0.,1.,0.,position[1]],
            [0.,0.,1.,position[2]],
            [0.,0.,0.,1.],
            ])

        rotation_matrix = GL.calculate_rotation_matrix(orientation[:3], orientation[3])

        # Invert the camera transformation to get the view matrix
        view_matrix = np.linalg.inv(camera_translation @ rotation_matrix)

        aspect_ratio = GL.width/GL.height
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

        rotation_matrix = GL.calculate_rotation_matrix(rotation[:3],rotation[3])

        # Correct the order of transformations
        transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix

        GL.transform_stack.append(transform_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        GL.transform_stack.pop()

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

        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
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
        
        # Processar cada face
        for face in faces:
            if len(face) < 3:
                continue  # É necessário pelo menos 3 vértices para formar um polígono

            # Triangularizar a face usando o método de fan (leque)
            v0 = face[0]
            for i in range(1, len(face) - 1):
                v1 = face[i]
                v2 = face[i + 1]

                # Obter as coordenadas dos vértices
                coords = []
                for vi in [v0, v1, v2]:
                    x = coord[3 * vi]
                    y = coord[3 * vi + 1]
                    z = coord[3 * vi + 2]
                    coords.extend([x, y, z])

                GL.triangleSet(coords, colors)


    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

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

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        value_changed = [0, 0, 1, 0]

        return value_changed

    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
