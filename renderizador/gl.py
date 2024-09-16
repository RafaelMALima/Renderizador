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

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    camera_transform_matrix = []
    transform_stack : list[NDArray] = []

    screen_matrix = []


    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.screen_matrix = np.array([[GL.width/2,0,0,GL.width/2],
                     [0,-GL.height/2,0,GL.height/2],
                     [0,0,1,0],
                     [0,0,0,1]])


    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        for i in range(0,len(point),2):
            pos_x = point[i]
            pos_y = point[i+1]
            cor = colors['emissiveColor']
            gpu.GPU.draw_pixel([int(pos_x), int(pos_y)], gpu.GPU.RGB8, [cor[0]*255,cor[1]*255,cor[2]*255])  # altera pixel (u, v, tipo, r, g, b)

        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

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
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
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


        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""

        if len(vertices) < 5:
            print("ERROR NO TRIANGLES SENT")
            return

        color = np.array(colors["emissiveColor"]) * 255

        for i in range(0, len(vertices), 6):
            tri = vertices[i: i + 6]

            if len(tri) != 6:
                return

            # Extrair as coordenadas dos vértices do triângulo
            p1 = [tri[0], tri[1]]
            p2 = [tri[2], tri[3]]
            p3 = [tri[4], tri[5]]

            xs = [p1[0], p2[0], p3[0]]
            ys = [p1[1], p2[1], p3[1]]

            # Calcular a caixa delimitadora (bounding box) do triângulo
            min_x = int(min(xs))
            max_x = int(max(xs))
            min_y = int(min(ys))
            max_y = int(max(ys))

            # Iterar sobre cada pixel dentro da bounding box
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    # Coordenadas do ponto central do pixel
                    px = x + 0.5
                    py = y + 0.5

                    # Cálculo das funções de linha para cada aresta do triângulo
                    L1 = (p2[1] - p1[1]) * px - (p2[0] - p1[0]) * py + p1[1] * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1])
                    L2 = (p3[1] - p2[1]) * px - (p3[0] - p2[0]) * py + p2[1] * (p3[0] - p2[0]) - p2[0] * (p3[1] - p2[1])
                    L3 = (p1[1] - p3[1]) * px - (p1[0] - p3[0]) * py + p3[1] * (p1[0] - p3[0]) - p3[0] * (p1[1] - p3[1])

                    # Verificar se o ponto está dentro do triângulo
                    if (L1 >= 0 and L2 >= 0 and L3 >= 0) or (L1 <= 0 and L2 <= 0 and L3 <= 0):
                        if 0 <= x < GL.width and 0 <= y < GL.height:
                            # Aqui você pode adicionar interpolação de cores ou texturas se necessário
                            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)



    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        for i in range(len(point) // 9):
            p = point[i*9:i*9+9]
            p_a : list[float] = [p[0], p[1], p[2]] # assigning dos pontos de uma forma mais fofa
            p_b : list[float] = [p[3], p[4], p[5]]
            p_c : list[float] = [p[6], p[7], p[8]] 

            triangle_matrix_no_transform = np.array([
                    [p_a[0], p_b[0], p_c[0]],
                    [p_a[1], p_b[1], p_c[1]],
                    [p_a[2], p_b[2], p_c[2]],
                    [1.,1.,1.]])


            #Apply all transforms in shape
            transform_matrix = np.identity(GL.transform_stack[0].shape[0])
            for matrix in GL.transform_stack:
                transform_matrix = transform_matrix @ matrix


            triangle_matrix_worldspace = transform_matrix@triangle_matrix_no_transform

            camera_transform_matrix = GL.camera_transform_matrix

            triangle_matrix_unnormalized = camera_transform_matrix@triangle_matrix_worldspace

            triangle_matrix = triangle_matrix_unnormalized / triangle_matrix_unnormalized[3][0]

            screen_matrix = GL.screen_matrix

            triangle_matrix_cameraspace = screen_matrix @ triangle_matrix

            triangle_array = np.asarray(triangle_matrix_cameraspace)

            triangle_points = [triangle_array[0][0],
                                triangle_array[1][0],
                                triangle_array[0][1],
                                triangle_array[1][1],
                                triangle_array[0][2],
                                triangle_array[1][2]]

            GL.triangleSet2D(triangle_points, colors)
            #print((p_a, p_b, p_c))

            # Exemplo de desenho de um pixel branco na coordenada 10, 10
            #gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        """print("Viewpoint : ", end='')
        print("position = {0} ".format(position), end='')
        print("orientation = {0} ".format(orientation), end='')
        print("fieldOfView = {0} ".format(fieldOfView))"""

        GL.position = position
        GL.orientation = orientation
        GL.fov = fieldOfView

        matrix_pos = np.matrix([
            [1.,0.,0.,-position[0]],
            [0.,1.,0.,-position[1]],
            [0.,0.,1.,-position[2]],
            [0.,0.,0.,1.],
            ])


        rotation_matrix = np.transpose(GL.calculate_rotation_matrix(orientation[:3], orientation[3]))
        
        view_matrix = rotation_matrix@matrix_pos

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

        GL.camera_transform_matrix = perspective_matrix@view_matrix

    @staticmethod
    def calculate_rotation_matrix(rot_v, theta):
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)

        q_i = rot_v[0] * sin
        q_j = rot_v[1] * sin
        q_k = rot_v[2] * sin
        q_r = cos


        rotation_matrix = np.array([[1-2*(q_j**2 - q_k**2), 2*(q_i*q_j - q_k*q_r), 2*(q_i*q_k + q_j*q_r), 0],
                    [2*(q_i*q_j + q_k*q_r), 1 - 2*(q_i**2 + q_k**2), 2*(q_i*q_k - q_i*q_r), 0],
                    [2*(q_i*q_k - q_j*q_r), 2*(q_j*q_k + q_i*q_r), 1-2*(q_i**2+q_j**2), 0],
                    [0,0,0,1]])
        return rotation_matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        '''
        print("Transform : ", end='')
        if translation:
            print("translation = {0} ".format(translation), end='') # imprime no terminal
        if scale:
            print("scale = {0} ".format(scale), end='') # imprime no terminal
        if rotation:
            print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        print("")
        '''
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

        transform_matrix = translation_matrix@scale_matrix@rotation_matrix

        GL.transform_stack.append(transform_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("Saindo de Transform")
        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        vertices = []                      
        print(point)
        for i in range(0,len(point),3):
            vertices.append([point[i],point[i+1],point[i+2]])

        coords_tri_set = []
        for i in range(len(vertices)-3):
            v_1 = vertices[i]
            v_2 = vertices[i+1]
            v_3 = vertices[i+2]
            for i in v_1+v_2+v_3:
                coords_tri_set.append(i)


        GL.triangleSet(coords_tri_set,colors)




    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
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
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um m                   texCoord, texCoordIndex, colors, current_texture):
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
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
