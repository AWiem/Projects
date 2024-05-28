import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FindAKeeperComponent } from './find-a-keeper.component';

describe('FindAKeeperComponent', () => {
  let component: FindAKeeperComponent;
  let fixture: ComponentFixture<FindAKeeperComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ FindAKeeperComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FindAKeeperComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
