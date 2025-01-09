#include "AVL.hpp"
#include <algorithm>

int AVLTree::height(AVLNode *node) { return node ? node->height : 0; }

// Helper function to get the balance factor of a node
int AVLTree::getBalance(AVLNode *node) {
  return node ? height(node->left) - height(node->right) : 0;
}

// Rotate right
AVLNode *AVLTree::rightRotate(AVLNode *y) {
  AVLNode *x = y->left;
  AVLNode *T2 = x->right;

  // Perform rotation
  x->right = y;
  y->left = T2;

  // Update heights
  y->height = std::max(height(y->left), height(y->right)) + 1;
  x->height = std::max(height(x->left), height(x->right)) + 1;

  return x; // New root
}

// Rotate left
AVLNode *AVLTree::leftRotate(AVLNode *x) {
  AVLNode *y = x->right;
  AVLNode *T2 = y->left;

  // Perform rotation
  y->left = x;
  x->right = T2;

  // Update heights
  x->height = std::max(height(x->left), height(x->right)) + 1;
  y->height = std::max(height(y->left), height(y->right)) + 1;

  return y; // New root
}

// Insert a new block (by start address) into the tree
AVLNode *AVLTree::insert(AVLNode *node, size_t start, size_t size) {
  if (!node)
    return new AVLNode(start, size);

  if (start < node->start_address)
    node->left = insert(node->left, start, size);
  else if (start > node->start_address)
    node->right = insert(node->right, start, size);
  else
    return node; // Duplicate blocks not allowed (unlikely)

  // Update height
  node->height = 1 + std::max(height(node->left), height(node->right));

  // Get balance factor and rotate if necessary
  int balance = getBalance(node);

  // Left Left Case
  if (balance > 1 && start < node->left->start_address)
    return rightRotate(node);

  // Right Right Case
  if (balance < -1 && start > node->right->start_address)
    return leftRotate(node);

  // Left Right Case
  if (balance > 1 && start > node->left->start_address) {
    node->left = leftRotate(node->left);
    return rightRotate(node);
  }

  // Right Left Case
  if (balance < -1 && start < node->right->start_address) {
    node->right = rightRotate(node->right);
    return leftRotate(node);
  }

  return node;
}

// Helper function to find the minimum node (used in deletion)
AVLNode *AVLTree::findMin(AVLNode *node) {
  AVLNode *current = node;
  while (current->left != nullptr)
    current = current->left;
  return current;
}

// Delete a block by start address
AVLNode *AVLTree::deleteNode(AVLNode *root, size_t start) {
  if (!root)
    return root;

  if (start < root->start_address)
    root->left = deleteNode(root->left, start);
  else if (start > root->start_address)
    root->right = deleteNode(root->right, start);
  else {
    // Node with only one child or no child
    if ((root->left == nullptr) || (root->right == nullptr)) {
      AVLNode *temp = root->left ? root->left : root->right;

      if (!temp) {
        temp = root;
        root = nullptr;
      } else
        *root = *temp;

      delete temp;
    } else {
      // Node with two children: Get the inorder successor
      AVLNode *temp = findMin(root->right);
      root->start_address = temp->start_address;
      root->size = temp->size;
      root->right = deleteNode(root->right, temp->start_address);
    }
  }

  if (!root)
    return root;

  // Update height
  root->height = 1 + std::max(height(root->left), height(root->right));

  // Get balance factor and rotate if necessary
  int balance = getBalance(root);

  // Left Left Case
  if (balance > 1 && getBalance(root->left) >= 0)
    return rightRotate(root);

  // Left Right Case
  if (balance > 1 && getBalance(root->left) < 0) {
    root->left = leftRotate(root->left);
    return rightRotate(root);
  }

  // Right Right Case
  if (balance < -1 && getBalance(root->right) <= 0)
    return leftRotate(root);

  // Right Left Case
  if (balance < -1 && getBalance(root->right) > 0) {
    root->right = rightRotate(root->right);
    return leftRotate(root);
  }

  return root;
}

AVLTree::AVLTree() : root(nullptr) {}

// Public method to insert a block into the AVL tree
void AVLTree::insert(size_t start, size_t size) {
  root = insert(root, start, size);
}

// Public method to delete a block from the AVL tree
void AVLTree::remove(size_t start) { root = deleteNode(root, start); }

// Method to find a block of at least the requested size
AVLNode *AVLTree::findFreeBlock(size_t size) {
  AVLNode *current = root;
  AVLNode *bestFit = nullptr;

  while (current) {
    if (current->size >= size) {
      bestFit = current;
      current = current->left; // Try to find a smaller suitable block
    } else {
      current = current->right;
    }
  }

  return bestFit;
}

// Method to find a block of at least the requested size
AVLNode *AVLTree::findInclusiveAddress(uintptr_t Addr, size_t size) {
  AVLNode *current = root;
  AVLNode *bestFit = nullptr;

  while (current) {
    if (current->size >= size) {
      bestFit = current;
      current = current->left; // Try to find a smaller suitable block
    } else {
      current = current->right;
    }
  }

  return bestFit;
}

AVLNode *AVLTree::getRoot() const { return root; }
