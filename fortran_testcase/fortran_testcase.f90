program HelloWorld
  integer :: i, N
  real :: p(100), v1(100), v2(100)
  
  write (*,*) 'Hello, world!'   ! This is an inline comment

  do i = 1,100
    v1(i) = i
    v2(i) = i
  end do

  !$omp target map(v1,v2,p)
  !$omp parallel do
  do i=1,100
    p(i) = v1(i) * v2(i)
  end do
  !$omp end target

  write (*,*) 'Hello, world!', p(9)
  
end program HelloWorld
