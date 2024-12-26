"""
Visualization Functions for PyTorch Tensors
===========================================

This module provides a collection of functions to visualize PyTorch tensors,
including feature maps, queries, attention maps, and position embeddings.
Each function is self-contained and easy to use, with detailed documentation
explaining how to use it and what it does.

Functions:
----------
- calculate_unique_values(tensor)
- plot_feature_map(tensor, channel_index, filename, show_colorbar=False)
- get_reference_points(spatial_shapes, valid_ratios, device)
- visualize_reference_points(reference_points, spatial_shapes)
- visualize_single_query(query, filename)
- visualize_queries_2d(queries, filename)
- visualize_attention_map(feature_map, sampling_locations, attention_weights, point_idx, level, filename)
- visualize_combined(feature_map, sampling_locations, attention_weights, query, point_idx, level, filename)
- visualize_position_embeddings(pos_embed, image_shape, num_rows, num_cols, filename)

Example Usage:
--------------
See the documentation of each function for example usage.

Note:
-----
Ensure you have the required libraries installed:
- torch
- numpy
- matplotlib

Import the module and use the functions as needed.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable


def calculate_unique_values(tensor):
    """
    Calculate unique values and their counts in a tensor.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        tuple(torch.Tensor, torch.Tensor): A tuple containing unique values and their counts.

    Example:
        unique_values, counts = calculate_unique_values(tensor)
    """
    unique_values, counts = torch.unique(tensor, return_counts=True)
    print("Unique values:", unique_values[:5])
    print("Counts of each unique value:", counts[:5])
    return unique_values, counts


def plot_feature_map(tensor, channel_index, filename, show_colorbar=False):
    """
    Plots a single feature map from a tensor and saves it to a file.

    Args:
        tensor (torch.Tensor): The input tensor of shape [B, C, H, W].
        channel_index (int): The channel index to visualize.
        filename (str): The filename to save the image.
        show_colorbar (bool): Whether to show a colorbar in the plot.

    Returns:
        None

    Example:
        plot_feature_map(x_rgb, 1000, "feature_map.png", show_colorbar=True)
    """
    # Extract the specific channel
    feature_map = tensor[:, channel_index, :, :]

    # Move the tensor to CPU and convert to numpy array
    image_np = feature_map.squeeze().cpu().numpy()  # shape [H, W]

    plt.imshow(image_np, cmap='viridis')
    plt.axis('off')
    if show_colorbar:
        plt.colorbar()
    plt.savefig(filename)
    plt.show()


def get_reference_points(spatial_shapes, valid_ratios, device):
    """
    Generates normalized reference points for Deformable DETR.

    Args:
        spatial_shapes (list of tuples): List of spatial shapes for each feature level 
                             (e.g., [(H1, W1), (H2, W2)] for two levels).
        valid_ratios (torch.Tensor): Tensor of shape (batch_size, num_levels, 2),
                          where each element is [height_ratio, width_ratio] 
                          to account for padding in the input. 
        device (str): Device to place tensors on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor of shape (batch_size, sum(H_i * W_i), 1, 2) containing
            normalized reference points (x, y) for all feature levels.

    Example:
        reference_points = get_reference_points(spatial_shapes, valid_ratios, device)
    """
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def visualize_reference_points(reference_points, spatial_shapes):
    """
    Visualizes reference points on a grid.

    Args:
        reference_points (torch.Tensor): Tensor of shape (batch_size, num_points, 1, 2)
            containing normalized reference points.
        spatial_shapes (list of tuples): List of spatial shapes [(H, W)].

    Returns:
        None

    Example:
        visualize_reference_points(reference_points, spatial_shapes)
    """
    # Assuming a single batch and level for simplicity:
    ref_pts = reference_points[0, :, 0, :].cpu().numpy()  # Get (x, y) points

    H, W = spatial_shapes[0]  # Assuming only one level for this example

    plt.figure(figsize=(8, 6))
    # Create a grid to visualize the feature map cells
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.xticks(np.arange(0, W, 1))
    plt.yticks(np.arange(0, H, 1))
    plt.grid(True)

    # Plot the reference points
    plt.scatter(ref_pts[:, 0] * W, ref_pts[:, 1] * H, s=20, color='red')
    plt.title("Reference Points Visualization")
    plt.show()


def visualize_single_query(query, filename):
    """
    Visualizes a single query vector as a heatmap.

    Args:
        query (torch.Tensor): A tensor of shape (256,) or (1, 256).
        filename (str): The filename to save the image.

    Returns:
        None

    Example:
        visualize_single_query(single_query, "query.png")
    """
    # Ensure query is 2D
    query = query.squeeze()
    if query.dim() == 1:
        query = query.unsqueeze(0)  # Shape (1, 256)

    # Plotting
    plt.figure(figsize=(10, 1))
    plt.imshow(query.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.yticks([])
    plt.xlabel('Dimension')
    plt.title('Visualization of a Single Query')
    plt.savefig(filename)
    plt.show()


def visualize_queries_2d(queries, filename):
    """
    Visualizes multiple queries as a grid of heatmaps.

    Args:
        queries (torch.Tensor): A tensor of shape (num_queries, num_dimensions).
        filename (str): The filename to save the image.

    Returns:
        None

    Example:
        visualize_queries_2d(object_queries, "queries.png")
    """
    queries = queries.detach().cpu().numpy()
    num_queries, num_channels = queries.shape

    # Adjust the number of rows and columns based on the number of queries
    num_cols = int(np.sqrt(num_queries))
    num_rows = int(np.ceil(num_queries / num_cols))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    fig.suptitle("Visualization of Object Queries", fontsize=16)

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i, query in enumerate(queries):
        ax = axes[i]
        query_2d = query.reshape(1, -1)
        im = ax.imshow(query_2d, aspect='auto', cmap='viridis')
        ax.set_title(f"Query {i+1}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for i in range(num_queries, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def visualize_attention_map(feature_map, sampling_locations, attention_weights, point_idx, level, filename):
    """
    Visualizes the attention map for a single query point.

    Args:
        feature_map (torch.Tensor): The feature map tensor of shape [H, W].
        sampling_locations (torch.Tensor): Sampling locations tensor.
        attention_weights (torch.Tensor): Attention weights tensor.
        point_idx (int): The index of the query point.
        level (int): The level of the feature map.
        filename (str): The filename to save the image.

    Returns:
        None

    Example:
        visualize_attention_map(feature_map, sampling_locations, attention_weights, point_idx, level, "attention_map.png")
    """
    # Convert tensors to numpy arrays
    feature_map = feature_map.cpu().numpy()
    sampling_locations = sampling_locations.cpu().numpy()
    attention_weights = attention_weights.cpu().numpy()

    H, W = feature_map.shape

    # Extract sampling locations and attention weights for the specified point and level
    point_sampling_locations = sampling_locations[0, point_idx, :, level, :, :]
    point_attention_weights = attention_weights[0, point_idx, :, level, :]

    plt.figure(figsize=(12, 10))
    plt.imshow(feature_map, cmap='viridis')

    # Define different shapes for each head
    marker_shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    # Plot the sampled points for each attention head
    n_heads = point_sampling_locations.shape[0]
    cmap = plt.cm.get_cmap('YlOrRd')

    for head in range(n_heads):
        head_locs = point_sampling_locations[head]
        head_weights = point_attention_weights[head]

        # Convert normalized coordinates to feature map coordinates
        head_locs[:, 0] *= W
        head_locs[:, 1] *= H

        # Filter out points outside the feature map
        valid_points = (head_locs[:, 0] >= 0) & (head_locs[:, 0] < W) & (head_locs[:, 1] >= 0) & (head_locs[:, 1] < H)

        plt.scatter(head_locs[valid_points, 0], head_locs[valid_points, 1],
                    c=head_weights[valid_points], cmap=cmap, s=100,
                    alpha=0.8, marker=marker_shapes[head % len(marker_shapes)], label=f'Head {head+1}')

    plt.title(f'Deformable Attention for Point {point_idx}')
    plt.axis('off')

    # Custom legend handler to remove color
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], marker=marker_shapes[i % len(marker_shapes)],
                               color='blue', linestyle='None') for i in range(n_heads)]
    new_handles.append(handles[-1])  # Add the query point handle

    # Add legend for each head with its shape
    plt.legend(new_handles, labels, loc='upper right', handler_map={tuple: HandlerTuple(ndivide=None)})

    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout()

    # Add a colorbar to show the attention weight scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Attention Weight', pad=0.01, orientation="horizontal")

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_combined(feature_map, sampling_locations, attention_weights, query, point_idx, level, filename):
    """
    Visualizes the attention map, feature map, and query vector together.

    Args:
        feature_map (torch.Tensor): Feature map tensor of shape [H, W].
        sampling_locations (torch.Tensor): Sampling locations tensor.
        attention_weights (torch.Tensor): Attention weights tensor.
        query (torch.Tensor): Query vector tensor of shape [256].
        point_idx (int): Index of the query point.
        level (int): Level of the feature map.
        filename (str): Filename to save the image.

    Returns:
        None

    Example:
        visualize_combined(feature_map, sampling_locations, attention_weights, query, point_idx, level, "combined.png")
    """
    # Convert tensors to numpy arrays
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    feature_map = to_numpy(feature_map)
    sampling_locations = to_numpy(sampling_locations)
    attention_weights = to_numpy(attention_weights)
    query = to_numpy(query).flatten()

    H, W = feature_map.shape

    # Extract sampling locations and attention weights for the specified point and level
    point_sampling_locations = sampling_locations[0, point_idx, :, level, :, :]
    point_attention_weights = attention_weights[0, point_idx, :, level, :]

    # Create the figure with a 2x2 grid
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.5, 1])

    # Decoder points visualization (left side)
    ax1 = fig.add_subplot(gs[:, 0])
    im1 = ax1.imshow(feature_map, cmap='viridis')

    # Define different shapes for each head
    marker_shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

    # Plot the sampled points for each attention head
    n_heads = point_sampling_locations.shape[0]
    cmap = plt.cm.get_cmap('YlOrRd')

    for head in range(n_heads):
        head_locs = point_sampling_locations[head]
        head_weights = point_attention_weights[head]

        # Convert normalized coordinates to feature map coordinates
        head_locs[:, 0] *= W
        head_locs[:, 1] *= H

        # Filter out points outside the feature map
        valid_points = (head_locs[:, 0] >= 0) & (head_locs[:, 0] < W) & (head_locs[:, 1] >= 0) & (head_locs[:, 1] < H)

        ax1.scatter(head_locs[valid_points, 0], head_locs[valid_points, 1],
                    c=head_weights[valid_points], cmap=cmap, s=100,
                    alpha=0.8, marker=marker_shapes[head % len(marker_shapes)], label=f'Head {head+1}')

    ax1.set_title('Deformable Attention Map')
    ax1.axis('off')

    # Custom legend handler to remove color
    handles, labels = ax1.get_legend_handles_labels()
    new_handles = [plt.Line2D([0], [0], marker=marker_shapes[i % len(marker_shapes)],
                               color='blue', linestyle='None') for i in range(n_heads)]
    new_handles.append(handles[-1])  # Add the query point handle

    # Add legend for each head with its shape
    ax1.legend(new_handles, labels, loc='upper right', handler_map={tuple: HandlerTuple(ndivide=None)})

    # Add a colorbar to show the attention weight scale
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax, label='Feature Map Value')

    # Bar plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(256), query)
    ax2.set_title("Bar Plot of Query Values")
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Value")
    ax2.set_xticks([0, 63, 127, 191, 255])

    # Query heatmap (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(query.reshape(1, -1), aspect='auto', cmap='viridis')
    ax3.set_title("Heatmap of Query Values")
    ax3.set_xlabel("Dimension")
    ax3.set_yticks([])
    ax3.set_xticks([0, 63, 127, 191, 255])

    # Add colorbar for the query heatmap
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("bottom", size="10%", pad=0.5)
    plt.colorbar(im3, cax=cax, orientation='horizontal', label='Query Value')

    plt.suptitle(f"Visualization of Query {point_idx}", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_position_embeddings(pos_embed, image_shape=(36, 63), num_rows=4, num_cols=4, filename="pos_embeddings.png"):
    """
    Visualizes multiple channels of position embeddings in 2D and 3D plots.

    Args:
        pos_embed (torch.Tensor): Position embedding tensor of shape (1, C, H, W).
        image_shape (tuple): The (height, width) of the image.
        num_rows (int): The number of rows for subplots.
        num_cols (int): The number of columns for subplots.
        filename (str): The filename to save the image.

    Returns:
        None

    Example:
        visualize_position_embeddings(pos_embed, image_shape=(36, 63), num_rows=2, num_cols=4)
    """
    pos_embed = pos_embed.squeeze(0).cpu().numpy()  # (C, H, W)
    height, width = image_shape
    num_channels = pos_embed.shape[0]

    # Create meshgrid for x, y coordinates 
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Calculate total subplots needed
    num_subplots = num_rows * num_cols

    # --- 3D Subplots ---
    fig_3d, axes_3d = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows),
                                 subplot_kw={'projection': '3d'})
    fig_3d.suptitle("Position Embeddings (3D)", fontsize=16)

    # --- 2D Heatmaps ---
    fig_2d, axes_2d = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    fig_2d.suptitle("Position Embeddings (2D)", fontsize=16)

    for channel_idx in range(min(num_channels, num_subplots)):
        embedding_channel = pos_embed[channel_idx]

        # Calculate subplot row and column index
        row_idx = channel_idx // num_cols
        col_idx = channel_idx % num_cols

        # --- 3D Plot ---
        ax_3d = axes_3d[row_idx, col_idx]
        ax_3d.plot_surface(X, Y, embedding_channel, cmap='viridis')
        ax_3d.set_xlabel("X (Width)")
        ax_3d.set_ylabel("Y (Height)")
        ax_3d.set_zlabel("Embedding Value")
        ax_3d.set_title(f"Channel {channel_idx}")

        # --- 2D Heatmap ---
        ax_2d = axes_2d[row_idx, col_idx]
        im = ax_2d.imshow(embedding_channel, cmap='viridis', aspect='auto')
        ax_2d.set_xlabel("X (Width)")
        ax_2d.set_ylabel("Y (Height)")
        ax_2d.set_title(f"Channel {channel_idx}")
        fig_2d.colorbar(im, ax=ax_2d)  # Add colorbar to 2D plot

    # Hide any unused subplots
    for i in range(num_channels, num_subplots):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axes_3d[row_idx, col_idx].axis('off')
        axes_2d[row_idx, col_idx].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.show()