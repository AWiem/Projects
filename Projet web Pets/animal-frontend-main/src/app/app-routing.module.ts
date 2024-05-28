import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LandingPageComponent } from './landing-page/landing-page.component';
import { StorePageComponent } from './store-page/store-page.component';
import { LoginComponent } from './log-in/log-in.component';
import { HowItWorksComponent } from './how-it-works/how-it-works.component';
import { ProfileCardComponent } from './profile-card/profile-card.component';
import { AdoptComponent } from './adopt/adopt.component';
import { ProductDetailsPageComponent } from './product-details-page/product-details-page.component';
import { BasketPageComponent } from './basket-page/basket-page.component';
import { RegisterComponent } from './register/register.component';
import { FindAKeeperComponent } from './find-a-keeper/find-a-keeper.component';
import { VeterinaryComponent } from './veterinary/veterinary.component';

const routes: Routes = [
  {path:"", redirectTo:"/landing", pathMatch: "full"},
  {path:"landing", component:LandingPageComponent},
  {path:"store", component:StorePageComponent},
  {path:"login", component:LoginComponent},
  {path:"howitworks", component:HowItWorksComponent},
  {path:"profile", component:ProfileCardComponent},
  {path:"adopt", component:AdoptComponent},
  { path: 'product/:id', component: ProductDetailsPageComponent },
  { path: 'basket', component: BasketPageComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'find-a-keeper', component: FindAKeeperComponent },
  { path: 'veterinary', component: VeterinaryComponent },

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
