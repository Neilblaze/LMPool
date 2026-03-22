import math

import torch
import torch.nn.functional as F


def npo2(len):
    """
    Returns the next power of 2 greater than or equal to the given length.

    For example:
        - If length is 5, returns 8 (2^3).
        - If length is 8, returns 8 (2^3).
        - If length is 9, returns 16 (2^4).

    Args:
        length (int): The input length.

    Returns:
        int: The next power of 2.
    """
    # Use math.ceil to round up log2(length), then compute the power of 2.
    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    Pads the input tensor along the length dimension to the next power of 2.

    Used to ensure the tensor length is a power of 2, which is required by
    certain parallel algorithms such as parallel scan.

    Args:
        X (Tensor): Input tensor of shape (B, L, D, N), where:
            - B: Batch size
            - L: Length dimension
            - D: Data dimension
            - N: Other dimension

    Returns:
        Y (Tensor): Padded tensor of shape (B, npo2(L), D, N), where:
            - npo2(L): The next power of 2 greater than or equal to L.
            - All other dimensions remain unchanged.
    """
    # Compute the next power of 2 for the length dimension L.
    len_npo2 = npo2(X.size(1))  # X.size(1) retrieves the value of the length dimension L.

    # Padding tuple format: (left, right, top, bottom, front, back).
    # Only the length dimension is padded; all other dimensions use 0.
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))  # (left, right, top, bottom, front, back)

    # Pad the tensor using F.pad.
    # Arguments:
    # - X: the input tensor to pad
    # - pad_tuple: the padding tuple
    # - "constant": constant-value padding
    # - 0: pad with zeros
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    """
    Implements the parallel scan operation, including forward and backward passes.
    Inherits from torch.autograd.Function to define a custom autograd function.
    """
    @staticmethod
    def pscan(A, X):
        """
        Forward parallel scan operation.

        Modifies X in-place to perform the parallel scan. More formally, X is filled with:
            H[t] = A[t] * H[t-1] + X[t]  where H[0] = 0
        These values are computed in parallel (ideally requiring 2*log2(T) sequential steps instead of T).

        Note:
            Only supports input lengths L that are a power of 2 (mainly for code clarity).

        Args:
            A (Tensor): Input tensor of shape (B, D, L, N).
            X (Tensor): Input tensor of shape (B, D, L, N).

        Returns:
            Tensor: Output tensor after parallel scan, same shape as X.
        """
        # Get the dimensions of the input tensors.
        B, D, L, _ = A.size()
        # Compute log2(L), the number of scan steps.
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        # Initialize Aa and Xa from input tensors A and X.
        Aa = A
        Xa = X

        # Iterate (num_steps - 2) times, progressively halving the number of elements processed.
        for _ in range(num_steps-2):
            # Current number of elements being processed.
            T = Xa.size(2)
            # Reshape Aa and Xa for parallel processing.
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # Update element 1 of Xa in parallel.
            # Xa[:, :, :, 1] += Aa[:, :, :, 1] * Xa[:, :, :, 0]
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            # Update element 1 of Aa in parallel.
            # Aa[:, :, :, 1] *= Aa[:, :, :, 0]
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            # Advance Aa and Xa to the current sub-tensors.
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # Handle the remaining 4, 2, or 1 nodes.
        if Xa.size(2) == 4:
            # Process element 1.
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            # Process element 3.
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            # Process element 1.
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        # Re-initialize Aa and Xa to the relevant slices of the input tensors.
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        # Update element 2 of Xa in parallel.
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        # Iterate (num_steps - 3) times, progressively expanding the number of elements processed.
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            # Reshape Aa and Xa for parallel processing.
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # Update the leading elements of Xa in parallel.
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            # Update the leading elements of Aa in parallel.
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        """
        Reverse parallel scan operation.

        Same as pscan above but traversed in the opposite direction.
        (Flipping the input, calling pscan, then flipping the output yields the same result as this function.)
        Used during the backward pass.

        Note:
            Only supports input lengths L that are a power of 2 (mainly for code clarity).

        Args:
            A (Tensor): Input tensor of shape (B, D, L, N).
            X (Tensor): Input tensor of shape (B, D, L, N).

        Returns:
            Tensor: Output tensor after reverse parallel scan, same shape as X.
        """
        # Get the dimensions of the input tensors.
        B, D, L, _ = A.size()
        # Compute log2(L), the number of scan steps.
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        # Initialize Aa and Xa from input tensors A and X.
        Aa = A
        Xa = X

        # Iterate (num_steps - 2) times, progressively halving the number of elements processed.
        for _ in range(num_steps-2):
            # Current number of elements being processed.
            T = Xa.size(2)
            # Reshape Aa and Xa for parallel processing.
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # Update element 0 of Xa in parallel.
            # Xa[:, :, :, 0] += Aa[:, :, :, 0] * Xa[:, :, :, 1]
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            # Update element 0 of Aa in parallel.
            # Aa[:, :, :, 0] *= Aa[:, :, :, 1]
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            # Advance Aa and Xa to the current sub-tensors.
            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # Handle the remaining 4, 2, or 1 nodes.
        if Xa.size(2) == 4:
            # Process element 2.
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            # Process element 0.
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            # Process element 0.
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        # Re-initialize Aa and Xa to the relevant slices of the input tensors.
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        # Update element 1 of Xa in parallel.
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        # Iterate (num_steps - 3) times, progressively expanding the number of elements processed.
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            # Reshape Aa and Xa for parallel processing.
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # Update the trailing elements of Xa in parallel.
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            # Update the trailing elements of Aa in parallel.
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation described above. Returns a new tensor.
        Prefer sequence lengths that are a power of 2 when possible.

        Args:
            A_in (Tensor): Input tensor of shape (B, L, D, N).
            X_in (Tensor): Input tensor of shape (B, L, D, N).

        Returns:
            H (Tensor): Output tensor of shape (B, L, D, N).
        """
        # Get the length dimension L.
        L = X_in.size(1)

        # Cloning required due to in-place operations.
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # Pad tensors (cloning happens implicitly).
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)

        # Prepare tensors.
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # Run parallel scan (modifies X in-place).
        PScan.pscan(A, X)

        # Save tensors for the backward pass.
        ctx.save_for_backward(A_in, X)

        # Slice [:, :L] to remove padding if applied.
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Propagates gradients from the output back to the inputs. Returns two new tensors.

        Args:
            ctx: A_in (Tensor): (B, L, D, N), X (Tensor): (B, D, L, N)
            grad_output_in (Tensor): (B, L, D, N)

        Returns:
            gradA (Tensor): (B, L, D, N), gradX (Tensor): (B, L, D, N)
        """
        # Retrieve saved tensors.
        A_in, X = ctx.saved_tensors

        # Get the length dimension L.
        L = grad_output_in.size(1)

        # Cloning required due to in-place operations.
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # The padding below will clone A_in.
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # Prepare tensors.
        grad_output = grad_output.transpose(2, 1) # (B, D, npo2(L), N)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # Run reverse parallel scan (modifies grad_output in-place).
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]

pscan = PScan.apply
