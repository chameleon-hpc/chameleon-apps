module test
 use iso_c_binding
 contains

  subroutine foo(n, y, arr) bind(c)
    integer, intent(in) :: n
    integer, intent(in) :: y
    type(c_ptr),value, intent(in) :: arr
  !  integer, dimension(50), intent(inout) :: arr
    integer :: i
    integer, pointer :: arr_ptr(:)

  ! write(*,*) 'cptr after',arr
   call c_f_pointer(arr, arr_ptr,[50]) 

    write(*,*) 'foo called', n, y
    do i=1, size(arr_ptr)
      write(*,*) i,arr_ptr(i)
    end do
  end subroutine

  subroutine printTest(n) bind(c)
    integer, intent(in) :: n
    write(*,*) 'task finished',n
  end subroutine 

end module 

program HelloWorld

use chameleon_lib
use test
use iso_c_binding

  integer :: i, N, ierr, y
  real :: p(100), v1(100), v2(100)
  type(c_ptr):: ctest, task_c_ptr
  type(map_entry), dimension(:), allocatable, target :: args
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

 ! write(*,*) 'c_ptr before', c_loc(arr)  

  ierr = chameleon_init()
  ierr = chameleon_determine_base_addresses(c_null_ptr)

! call foo(i, y, arr) 
  task_c_ptr = chameleon_create_task(foo, 3, args)  
  ierr = chameleon_set_callback_task_finish(task_c_ptr, c_funloc(printTest), c_loc(i))
  ierr = chameleon_add_task_fortran(task_c_ptr)
 
  ierr = chameleon_distributed_taskwait(0)

  ierr = chameleon_finalize()
  
end program HelloWorld
