nums = [4, 3, 2, 0, 6, 7, 5,1,1,1,1]

def find_missing_number(nums):
    x = []
    nums =  sorted(nums)
    if nums == []:
        return(0)
    
    for i in range(len(nums)):
        if nums[i] == 0:
            x.append(nums[i])
    return(x)
    
print(find_missing_number(nums))