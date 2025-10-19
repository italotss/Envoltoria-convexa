import pygame
import numpy as np
import math

#Change the colors of the lines and points here
point_color = (249, 127, 81)
hull_color = (27, 156, 252)
line_color = (37, 204, 247)

def direction(a, b, c):
    #(b-a) x (c-a)
    #The points are according to python, hence check for negative
    return (((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) < 0)


def create_circle_array(position, radius, num_points):
    if num_points < 3:
        raise ValueError("num_points must be at least 3 to form a circle")
        return 0
    else:
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = position[0] + radius * math.cos(angle)
            y = position[1] + radius * math.sin(angle)
            points.append((x, y))
        return points


#this is bad code but works right now. will be completely rewritten later
def create_triangle_array(position, num_points):  # creates a triangle to the input's 3 points and places num_points along its perimeter
    if num_points < 3:
        raise ValueError("num_points must be at least 3 to form a triangle")
    points = []
    last_3_0 = None
    last_3_1 = None
    last_2_0 = None
    last_2_1 = None
    last_1_0 = None
    last_1_1 = None
    counter_3 = 1
    counter_2 = 1
    counter_1 = 1  
    for i in range(num_points):  # additional points to be placed along the triangle's perimeter
        if i % 3 == 0:  # between first and third point
            if last_3_0 is None:  # initialize the first midpoints
                x = (position[0][0] + position[2][0]) / 2
                y = (position[0][1] + position[2][1]) / 2
                last_3_0 = (x, y)
                points.append((x, y))
                continue
            elif last_3_1 is None:
                x = (last_3_0[0] + position[0][0]) / 2
                y = (last_3_0[1] + position[0][1]) / 2  # FIX: use position[0][1]
                last_3_1 = (x, y)
                points.append((x, y))
                continue
            if counter_3 > 0:  # alternates between the two midpoints
                x = (position[0][0] + last_3_0[0]) / 2
                y = (position[0][1] + last_3_0[1]) / 2
                last_3_0 = (x, y)
                points.append((x, y))
                counter_3 *= -1
                continue
            else:
                x = (last_3_0[0] + position[2][0]) / 2
                y = (last_3_0[1] + position[2][1]) / 2
                last_3_1 = (x, y)
                points.append((x, y))
                counter_3 *= -1
                continue

        elif i % 2 != 0:  # between second and third point
            if last_2_0 is None:
                x = (position[1][0] + position[2][0]) / 2
                y = (position[1][1] + position[2][1]) / 2
                last_2_0 = (x, y)
                points.append((x, y))
                continue
            elif last_2_1 is None:
                x = (last_2_0[0] + position[1][0]) / 2
                y = (last_2_0[1] + position[1][1]) / 2  # FIX: use position[1][1]
                last_2_1 = (x, y)
                points.append((x, y))
                continue
            if counter_2 > 0:
                x = (position[1][0] + last_2_0[0]) / 2
                y = (position[1][1] + last_2_0[1]) / 2
                last_2_0 = (x, y)
                points.append((x, y))
                counter_2 *= -1
                continue
            else:
                x = (last_2_0[0] + position[2][0]) / 2
                y = (last_2_0[1] + position[2][1]) / 2
                last_2_1 = (x, y)
                points.append((x, y))
                counter_2 *= -1
                continue
        
        else:  # between first and second point
            if last_1_0 is None:
                x = (position[0][0] + position[1][0]) / 2
                y = (position[0][1] + position[1][1]) / 2
                last_1_0 = (x, y)
                points.append((x, y))
                continue
            elif last_1_1 is None:
                x = (last_1_0[0] + position[0][0]) / 2
                y = (last_1_0[1] + position[0][1]) / 2  # FIX: use position[0][1]
                last_1_1 = (x, y)
                points.append((x, y))
                continue
            if counter_1 > 0:
                x = (position[0][0] + last_1_0[0]) / 2
                y = (position[0][1] + last_1_0[1]) / 2
                last_1_0 = (x, y)
                points.append((x, y))
                counter_1 *= -1
                continue
            else:
                x = (last_1_0[0] + position[1][0]) / 2
                y = (last_1_0[1] + position[1][1]) / 2
                last_1_1 = (x, y)
                points.append((x, y))
                counter_1 *= -1
                continue

    # ---- Perimeter ordering (A->B->C->A), include vertices so corners are exact ----
    A, B, C = [tuple(p) for p in position]
    edges = [(A, B), (B, C), (C, A)]
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in edges]
    perim = sum(lengths)
    eps = 1e-6 * max(1.0, perim)

    def project_param(p, a, b):
        ax, ay = a; bx, by = b; px, py = p
        vx, vy = (bx - ax), (by - ay)
        wx, wy = (px - ax), (py - ay)
        L2 = vx * vx + vy * vy
        if L2 == 0:
            return None, None, None
        t = (wx * vx + wy * vy) / L2
        cross = vx * wy - vy * wx
        dist = abs(cross) / math.sqrt(L2)
        return t, cross, dist

    ordered = []
    # Add the 3 exact corners so the polygon follows the perimeter
    ordered.append((0.0, A))
    ordered.append((lengths[0], B))
    ordered.append((lengths[0] + lengths[1], C))

    for p in points:
        placed = False
        s_accum = 0.0
        for (a, b), L in zip(edges, lengths):
            t, cross, dist = project_param(p, a, b)
            if L == 0:
                s_accum += L
                continue
            if abs(cross) <= eps and -1e-6 <= t <= 1 + 1e-6:
                t_clamped = min(max(t, 0.0), 1.0)
                s = s_accum + t_clamped * L
                ordered.append((s, p))
                placed = True
                break
            s_accum += L
        if not placed:
            # Fallback: nearest edge
            best = None
            s_accum = 0.0
            for (a, b), L in zip(edges, lengths):
                t, _, dist = project_param(p, a, b)
                t_clamped = 0.0 if t is None else min(max(t, 0.0), 1.0)
                cand_s = s_accum + t_clamped * L
                if best is None or dist < best[0]:
                    best = (dist, cand_s)
                s_accum += L
            ordered.append((best[1], p))

    ordered.sort(key=lambda sp: sp[0] % perim)
    points = [p for _, p in ordered]
    return points

def compute_convex_hull(coords):

    # Compute convex hull using gift wrapping algorithm. Returns list of hull points in order.

    if len(coords) < 3:
        return coords.copy()
    
    hull = []
    
    # Find leftmost point
    leftmost_x = min([xcoord[0] for xcoord in coords])
    leftmost_point = [t for t in coords if leftmost_x == t[0]][0]
    
    current_point = leftmost_point
    
    while True:
        hull.append(current_point)
        next_point = coords[(coords.index(current_point) + 1) % len(coords)]
        
        for check_point in coords:
            if direction(current_point, next_point, check_point):
                next_point = check_point
        
        current_point = next_point
        
        if current_point == hull[0]:
            break
    
    return hull


def draw_convex_hull(window, coords):

    if len(coords) < 3:
        return
    hull = compute_convex_hull(coords)
    
    # Draw hull points
    for point in hull:
        pygame.draw.circle(window, hull_color, point, 4)
    
    # Draw hull edges
    for i in range(len(hull)):
        next_i = (i + 1) % len(hull)
        pygame.draw.line(window, line_color, hull[i], hull[next_i], 4)


def main():    
    pygame.init()
    window = pygame.display.set_mode((1000, 1000))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Convex Hull")
    
    # n is the number of points
    n = 30
    points = np.random.randint(25, 950, size=2*n)
    
    # Arrays for the coordinates
    coords = []

    for i in range(0, len(points)-1, 2):
        coords.append([points[i], points[i+1]])
    
    
    # Add initial random points
    random_coords = np.random.randint(25, 950, 14)

    triangle_array_1 = create_triangle_array((
    [random_coords[0], random_coords[1]], 
    [random_coords[2], random_coords[3]], 
    [random_coords[4], random_coords[5]]
    ), np.random.randint(3, 7))
    coords.extend(triangle_array_1)

    triangle_array_2 = create_triangle_array((
    [random_coords[6], random_coords[7]],
    [random_coords[8], random_coords[9]],
    [random_coords[10], random_coords[11]]
    ), np.random.randint(3, 7))
    coords.extend(triangle_array_2)

    circle_array = create_circle_array((random_coords[12:14]), 60, np.random.randint(5, 15))
    coords.extend(circle_array)

    # Main loop
    running = True
    while running:
        window.fill((0, 0, 0))  # Clear screen
        
        # Draw all points
        for point in coords:
            pygame.draw.circle(window, point_color, point, 4)
        
        pygame.draw.polygon(window, point_color, triangle_array_1, 1)
        pygame.draw.polygon(window, point_color, triangle_array_2, 1)
        pygame.draw.polygon(window, point_color, circle_array, 1)

        # Draw convex hull
        if len(coords) >= 3:
            draw_convex_hull(window, coords)
        
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                # Add new point and update hull
                coords.append(list(event.pos))
                
            if event.type == pygame.QUIT:
                running = False
   
    pygame.quit()

if __name__ == "__main__":
    main()