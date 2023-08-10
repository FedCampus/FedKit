void assertEqual(left, right) {
  if (left != right) {
    throw Exception('Assertion failed: $left != $right');
  }
}
