module test
 contains

  subroutine foo(n, y, arr) bind(c)
    integer, intent(in) :: n
    integer, intent(in) :: y
    integer, dimension(50), intent(inout) :: arr
    integer :: i
 
    write(*,*) 'foo called', n, y
    do i=1, size(arr)
      write(*,*) i,arr(i)
    end do
  end subroutine
end module 

program HelloWorld

use chameleon_lib
use test
use iso_c_binding

  integer :: i, N, ierr, y
  real :: p(100), v1(100), v2(100)
  type(c_ptr):: ctest
  type(map_entry), dimension(:), allocatable :: args
  integer, dimension(:), allocatable :: arr

  allocate(arr(1:50))

  do i=1,50
    arr(i) = i
  end do

  allocate(args(1:2))

  i = 42
  y = 43

  args(1)%valptr = c_loc(i)
  args(1)%size = sizeof(i)
  args(1)%type = 3

  args(2)%valptr = c_loc(y)
  args(2)%size = sizeof(y)
  args(2)%type = 3

  args(3)%valptr = c_loc(arr)
  args(3)%size = sizeof(arr)
  args(3)%type = 3

  ierr = chameleon_init()
  ierr = chameleon_determine_base_addresses(c_null_ptr)

! call foo(i, y, arr) 

  ierr = chameleon_add_task_manual(foo, 3, args)

  ierr = chameleon_distributed_taskwait(0)

  ierr = chameleon_finalize()
  
end program HelloWorld
