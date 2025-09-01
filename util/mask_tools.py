import numpy as np
import torch
import random

mask_types=["left", "right", "top", "bottom", "center", "surrounding", "random_shape", "random_stroke", "all"]

def _bezier_curve(p0, p1, p2, num_points=100):
    """Generate points for a quadratic Bezier curve"""
    t = np.linspace(0, 1, num_points)
    # Quadratic Bezier curve formula: B(t) = (1-t)^2 * P0 + 2t(1-t) * P1 + t^2 * P2
    x = (1-t)**2 * p0[0] + 2*t*(1-t) * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*t*(1-t) * p1[1] + t**2 * p2[1]
    return np.stack([x, y], axis=1)

def generate_random_mask(w, h, mask_type):
    mask = torch.zeros((1, 1, h, w))
    
    if mask_type == "all":
        mask.fill_(1)
    elif mask_type == "random_stroke":
        # Randomly select the area ratio to mask (10%-40%)
        area_ratio = random.uniform(0.1, 0.4)
        
        # Generate 1-3 random lines
        num_strokes = random.randint(1, 3)
        
        for _ in range(num_strokes):
            # Randomly generate the start and end points of the line
            start_x = random.randint(0, w-1)
            start_y = random.randint(0, h-1)
            end_x = random.randint(0, w-1)
            end_y = random.randint(0, h-1)
            
            # Randomly generate control points to make the line more natural
            ctrl_x = random.randint(min(start_x, end_x), max(start_x, end_x))
            ctrl_y = random.randint(min(start_y, end_y), max(start_y, end_y))
            
            # Generate Bezier curve points
            curve_points = _bezier_curve(
                np.array([start_x, start_y]),
                np.array([ctrl_x, ctrl_y]),
                np.array([end_x, end_y]),
                num_points=100
            )
            
            # Randomly generate line thickness (2%-8% of the shorter side of the image)
            stroke_width = random.uniform(0.02, 0.08) * min(w, h)
            
            # Create thick lines around the curve points
            for point in curve_points:
                x, y = point.astype(int)
                # Ensure coordinates are within the image range
                if 0 <= x < w and 0 <= y < h:
                    # Create circular brush
                    for i in range(max(0, int(y-stroke_width)), min(h, int(y+stroke_width))):
                        for j in range(max(0, int(x-stroke_width)), min(w, int(x+stroke_width))):
                            # Calculate distance to center point
                            if ((i-y)**2 + (j-x)**2) <= stroke_width**2:
                                mask[0, 0, i, j] = 1
    
    elif mask_type == "random_shape":
        # Randomly select the area ratio to mask (20%-80%)
        area_ratio = random.uniform(0.2, 0.8)
        
        # Generate random control point count (4-8 points)
        num_control_points = random.randint(4, 8)
        
        # Generate center point
        center_x = w // 2
        center_y = h // 2
        
        # Calculate base radius (based on target area)
        base_radius = np.sqrt(w * h * area_ratio / np.pi)
        
        # Generate control points
        control_points = []
        angles = np.linspace(0, 2*np.pi, num_control_points, endpoint=False)
        for angle in angles:
            # Randomly within 60%-140% of the base radius
            radius = base_radius * random.uniform(0.6, 1.4)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            control_points.append([x, y])
        
        # Ensure closed curve
        control_points.append(control_points[0])
        
        # Generate all Bezier curve segment points
        curve_points = []
        for i in range(len(control_points)-1):
            # Generate control points between two points, adding some random offset
            p0 = control_points[i]
            p2 = control_points[i+1]
            mid_x = (p0[0] + p2[0]) / 2
            mid_y = (p0[1] + p2[1]) / 2
            offset = base_radius * random.uniform(-0.5, 0.5)
            angle = np.arctan2(p2[1]-p0[1], p2[0]-p0[0]) + np.pi/2
            p1 = [
                mid_x + offset * np.cos(angle),
                mid_y + offset * np.sin(angle)
            ]
            
            # Generate Bezier curve points
            curve_segment = _bezier_curve(
                np.array(p0),
                np.array(p1),
                np.array(p2),
                num_points=50
            )
            curve_points.extend(curve_segment)
        
        # Convert to numpy array
        curve_points = np.array(curve_points)
        
        # Create polygon mask
        xx, yy = np.mgrid[:h, :w]
        points = np.vstack((xx.ravel(), yy.ravel())).T
        
        # Use matplotlib's path module to determine if points are inside the polygon
        from matplotlib.path import Path
        polygon_path = Path(curve_points)
        mask_array = polygon_path.contains_points(points)
        mask_array = mask_array.reshape(h, w)
        
        # Convert to torch tensor and add dimension
        mask = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)
        
    else:
        # Original mask generation logic
        area_ratio = random.uniform(0.2, 0.8)
        
        if mask_type == "left":
            mask_width = int(w * area_ratio)
            mask[:, :, :, :mask_width] = 1
        elif mask_type == "right":
            mask_width = int(w * area_ratio)
            mask[:, :, :, -mask_width:] = 1
        elif mask_type == "top":
            mask_height = int(h * area_ratio)
            mask[:, :, :mask_height, :] = 1
        elif mask_type == "bottom":
            mask_height = int(h * area_ratio)
            mask[:, :, -mask_height:, :] = 1
        elif mask_type == "center":
            center_w = int(w * np.sqrt(area_ratio))
            center_h = int(h * np.sqrt(area_ratio))
            
            # Calculate starting position to ensure center alignment
            start_w = (w - center_w) // 2
            start_h = (h - center_h) // 2
            
            mask[:, :, start_h:start_h+center_h, start_w:start_w+center_w] = 1
        
        elif mask_type == "surrounding":
            center_w = int(w * (1 - np.sqrt(area_ratio)))
            center_h = int(h * (1 - np.sqrt(area_ratio)))
            
            # Calculate starting position to ensure center alignment
            start_w = (w - center_w) // 2
            start_h = (h - center_h) // 2
            
            # Set surrounding mask (everything outside the center area is 1)
            mask.fill_(1)
            mask[:, :, start_h:start_h+center_h, start_w:start_w+center_w] = 0
        
    return mask

def test_masks():
    """Test different types of masks"""
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Create folder to save results
    save_dir = "mask_test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test image
    test_img_path = "flux-dev.png"  # Please replace with actual test image path
    img = Image.open(test_img_path)
    w, h = img.size
    
    # All mask types
    mask_types = ["left", "right", "top", "bottom", "center", "surrounding", "random_shape", "random_stroke"]
    
    # Test each mask type 5 times
    for mask_type in mask_types:
        for i in range(5):
            # Generate mask
            mask = generate_random_mask(w, h, mask_type)
            
            # Convert mask to PIL image
            mask_np = mask[0, 0].numpy()
            mask_img = Image.fromarray((mask_np * 255).astype('uint8'))
            
            # Create image combination for display
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Show original image
            ax1.imshow(img)
            ax1.set_title('Original')
            ax1.axis('off')
            
            # Show mask
            ax2.imshow(mask_img, cmap='gray')
            ax2.set_title(f'Mask ({mask_type})')
            ax2.axis('off')
            
            # Show image with mask applied
            masked_img = img.copy()
            masked_img = np.array(masked_img)
            mask_np = mask[0, 0].numpy()[:, :, np.newaxis]
            masked_img = masked_img * (1 - mask_np) + 255 * mask_np
            masked_img = masked_img.astype('uint8')
            
            ax3.imshow(masked_img)
            ax3.set_title('Masked')
            ax3.axis('off')
            
            # Save results
            plt.savefig(os.path.join(save_dir, f'{mask_type}_test_{i}.png'))
            plt.close()
            
    print(f"Test results have been saved to the {save_dir} folder")

if __name__ == "__main__":
    test_masks()
