import numpy as np
import cv2

def get_dominant_color(frame, bbox):
    """Extract the dominant color from the given bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    pixels = cropped.reshape(-1, 3)
    dominant_color = np.mean(pixels, axis=0)
    return dominant_color

def group_players_by_color(player_colors, k=2):
    """Group players into teams based on jersey color using K-means clustering."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(player_colors)
    centers = kmeans.cluster_centers_
    return labels, centers

def assign_goalkeeper_to_team(goalkeeper_color, team_centers):
    """Assign the goalkeeper to the closest team based on color similarity."""
    distances = [np.linalg.norm(goalkeeper_color - center) for center in team_centers]
    closest_team = np.argmin(distances)
    return closest_team
