def boundary(pt):
  max_x = max_y = 0
  min_x = min_y = 1e5
  for ind in range(len(pt)):
    max_x = max(max_x,pt[ind][0])
    max_y = max(max_y,pt[ind][1])
    min_x = min(min_x,pt[ind][0])
    min_y = min(min_y,pt[ind][1])
  return max_x, min_x, max_y, min_y