def fb(n):
    if n == 1 or n == 2:
        return 1
    return fb(n-1) + fb(n-2)

memo = {}
def fb_with_memo(memo, n):
    if n == 1 or n == 2:
        return 1
    if n in memo:
        return memo[n]
    else:
        memo[n] = fb_with_memo(memo, n-1) + fb_with_memo(memo, n-2)
        return memo[n]

def fb_iter(n):
    memo = [0]*(n+1)
    memo[1] = 1
    memo[2] = 1
    for i in range(3,n+1):
        print(i)
        memo[i] = memo[i-1]+memo[i-2]
    return memo[-1]

def fb_iter_pro(n):
    if n == 1 or n == 2:
        return 1
    pre = 1
    cur = 1
    for i in range(3, n+1):
        sum = cur + pre
        pre = cur
        cur = sum
    return cur
import torch
a = [[1,2,3,4],[1,2,3,4]]
ta = torch.FloatTensor(a)
ta = ta.unsqueeze(0)
print(ta)
ta.expand(2,2,2)
print(a)