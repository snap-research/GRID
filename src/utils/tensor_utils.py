from typing import Optional, Tuple, Union

import torch

from src.utils.file_utils import open_local_or_remote


def locations_to_index_tuple(locations: torch.Tensor, num_dims: int = 2) -> Tuple:
    """
    Convert a tensor of locations to a tuple of index tensors for advanced indexing.

    Args:
        locations (torch.Tensor): A tensor of shape `[L, D]` where `L` is the number of
            locations and `D >= num_dims`.
        num_dims (int): The number of dimensions to extract. The first num_dims columns of
            the locations tensor are used. We explicitly specify this to make the
            function call traceable.

    Returns:
        Tuple: A tuple of `num_dims` tensors, each of shape `[L]` representing the
            indices for one dimension.

    Example:
        >>> locations = torch.tensor([[0, 10], [1, 20], [2, 5]])
        >>> locations_to_index_tuple(locations, num_dims=2)
        (tensor([0, 1, 2]), tensor([10, 20,  5]))

        >>> locations = torch.tensor([[0, 10], [1, 20], [2, 5]])
        >>> locations_to_index_tuple(locations, num_dims=1)
        (tensor([0, 1, 2]))
    """
    return tuple(locations[:, i] for i in range(num_dims))


def extract_locations(
    data: torch.tensor, locations: torch.tensor, num_dims: int = 2
) -> torch.tensor:
    """
    Extracts the elements from a tensor at the specified indices.

    Args:
        data (torch.tensor): The input tensor of N dimensions from which to extract elements.
        locations (torch.tensor): Tensor of shape [L, D] where L is the number of
        elements where each D dimensional row reprecents the first D dimensions
        of the data tensor to extract.
        num_dims (int): The number of dimensions to extract. The first num_dims columns of
        the locations tensors are used. We need to specify to make this function call traceable.

    Returns:
        torch.tensor: A tensor of shape [L,...] with total N-num_dims+1 dimensions
        containing the extracted elements.

    Example:
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> locations = torch.tensor([[0, 1], [1, 2]])
        >>> extract_locations(data, locations, num_dims=2)
        tensor([2, 6]) # (Index 0,1 gives 2 (First row, second column); Index 1,2 gives 6 (Second row, third column))

        >>> locations = torch.tensor([[0, 1], [2, 0]])
        >>> extract_locations(data, locations, num_dims=1)
        tensor([[1, 2, 3], [7, 8, 9]])
        # (num_dims = 1 implies we are extracting based on the first dimension only.
        # Thus, we get the first row (from [0,1] as 1 is ignored) and the third row
        # (from [2,0] as 0 is ignored) of the data tensor.
    """

    # Separate the locations for each of the first D dimensions
    index_tuple = locations_to_index_tuple(locations=locations, num_dims=num_dims)

    # Use indexing with a tuple of index tensors
    extracted_values = data[index_tuple]

    return extracted_values


def merge_list_of_keyed_tensors_to_single_tensor(
    data: list[dict[str, torch.Tensor]],
    index_key: str,
    value_key: str,
) -> torch.Tensor:
    """
    Converts a list of dictionaries of id to tensors into a single tensor by squeezing
    the tensors along the specified index key.
    e.g.,
    data = [
    [
        {'user_id': 123,
        {'semantic_id': torch.tensor([21, 32, 124]),
        other features.....}
        ],
    [
        {'user_id': 456,
        {'semantic_id': torch.tensor([11, 22, 33]),
        other features.....}
    ]
    output:
    tensor([...,
            [21, 32, 124], # row 123
            ...,
            [11, 22, 33], # row 456
            ...
            ])

    Args:
        data (list[dict[str, torch.Tensor]]): A list of dictionaries where each dictionary
            contains an index key and a value key.
        index_key (str): The key in the dictionary that contains the index for each row.
        value_key (str): The key in the dictionary that contains the tensor to be merged.
    """
    batch_size = len(data)
    dimensions = torch.tensor(data[0][value_key]).size()
    output_tensor = torch.zeros((batch_size, *dimensions))
    for row in data:
        index = row[index_key]
        value = row[value_key]
        if index < batch_size:
            output_tensor[index] = torch.tensor(value)
        else:
            raise IndexError(
                f"Index {index} out of bounds for batch size {batch_size}."
            )
    return output_tensor


def deduplicate_rows_in_tensor(
    file_path: Optional[str] = None, return_tensor: bool = False
) -> Union[None, torch.Tensor]:
    """
    Identifies and de-duplicate repeated rows in a PyTorch tensor.
    Rows that are not duplicated will have a new column with value 0,
    while rows that are duplicated will have a new column indicating the number of duplicates from 0 to N-1
    where N is the number of duplicates for that row.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
        return_tensor: If True, returns the modified tensor; otherwise, saves it to the file.
    Returns:
        If return_tensor is True, returns the modified tensor with a new column indicating
        the number of duplicates for each row. If False, saves the modified tensor to the file.
    """
    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))
    assert len(data.size()) == 2, "Input data must be a 2D PyTorch tensor."

    # Use torch.unique to get unique rows and their inverse indices
    unique_rows, inverse_indices, counts = torch.unique(
        data, dim=0, return_inverse=True, return_counts=True
    )

    output_indices = torch.zeros_like(inverse_indices)

    # Sort indices by their cluster assignment to group duplicates together
    sorted_idx = torch.argsort(inverse_indices, stable=True)

    # For each position in sorted order, check if it's same cluster as previous
    # and increment counter accordingly
    for i in range(1, len(sorted_idx)):
        curr_cluster = inverse_indices[sorted_idx[i]]
        prev_cluster = inverse_indices[sorted_idx[i - 1]]
        if curr_cluster == prev_cluster:
            output_indices[sorted_idx[i]] = output_indices[sorted_idx[i - 1]] + 1

    # Concatenate the duplicate indicator column to the original data
    result = torch.cat((data, output_indices.unsqueeze(1)), dim=1).long()
    if return_tensor:
        return result
    else:
        # Save the result to a file
        torch.save(result, file_path)
        return None


def transpose_tensor_from_file(
    file_path: Optional[str] = None,
    return_tensor: bool = False,
    dim1: int = -2,
    dim2: int = -1,
) -> Union[None, torch.Tensor]:
    """
    Transposes a PyTorch tensor from a file accoridng to designated dimensions.

    Args:
        file_path: Optional; Path to a file containing the tensor data.
        return_tensor: If True, returns the modified tensor; otherwise, saves it to the file.
        dim1: The first dimension to transpose (default: -2).
        dim2: The second dimension to transpose (default: -1).
    Returns:
        If return_tensor is True, returns the modified tensor. If False, saves the modified tensor to the file.
    """
    if not file_path.endswith(".pt"):
        return None
    data = torch.load(open_local_or_remote(file_path, mode="rb"))

    # Transpose the tensor
    result = data.transpose(dim1, dim2)
    if return_tensor:
        return result
    else:
        # Save the result to a file
        torch.save(result, file_path)
        return None
